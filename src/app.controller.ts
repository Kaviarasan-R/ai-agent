import {
  BadRequestException,
  Body,
  Controller,
  Get,
  OnModuleInit,
  Post,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';

import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from '@langchain/google-genai';

import { PineconeStore } from '@langchain/pinecone';
import { Pinecone as PineconeClient } from '@pinecone-database/pinecone';

import { CSVLoader } from '@langchain/community/document_loaders/fs/csv';

import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import { FileInterceptor } from '@nestjs/platform-express';
import { AppService } from './app.service';

@Controller()
export class AppController implements OnModuleInit {
  private writeFile = promisify(fs.writeFile);
  private unlink = promisify(fs.unlink);
  private embeddings: GoogleGenerativeAIEmbeddings;
  private model: ChatGoogleGenerativeAI;
  private vectorStore: PineconeStore;
  private isInitialized = false;

  constructor(private readonly appService: AppService) {}

  async onModuleInit() {
    try {
      console.log('Initializing AI components...');

      this.embeddings = new GoogleGenerativeAIEmbeddings({
        model: 'gemini-embedding-001',
      });

      this.model = new ChatGoogleGenerativeAI({
        model: 'gemini-2.0-flash',
        temperature: 0.6,
      });

      const pinecone = new PineconeClient();
      const pineconeIndex = pinecone.Index('test');

      this.vectorStore = new PineconeStore(this.embeddings, {
        pineconeIndex,
        maxConcurrency: 5,
      });

      this.isInitialized = true;
      console.log('✓ AI components initialized successfully');
    } catch (error) {
      console.error('Failed to initialize AI components:', error);
      throw error;
    }
  }

  private async ensureInitialized() {
    if (!this.isInitialized) {
      throw new Error('AI components not initialized yet');
    }
  }

  @Post('sync')
  @UseInterceptors(FileInterceptor('file'))
  async sync(@UploadedFile() file: any) {
    if (!file) {
      throw new BadRequestException('No CSV file provided');
    }

    if (file.mimetype !== 'text/csv' && !file.originalname.endsWith('.csv')) {
      throw new BadRequestException('File must be a CSV');
    }

    await this.ensureInitialized();

    let tempFilePath: string | null = null;

    try {
      tempFilePath = path.join(
        __dirname,
        '../',
        `temp_${Date.now()}_${file.originalname}`,
      );
      await this.writeFile(tempFilePath, file.buffer);

      const loader = new CSVLoader(tempFilePath);
      const docs = await loader.load();

      console.log(`Loaded ${docs.length} documents from CSV`);

      const batchSize = 50;
      const totalBatches = Math.ceil(docs.length / batchSize);
      let processedCount = 0;
      const results: any = [];

      if (docs.length > 0) {
        const testEmbedding = await this.embeddings.embedQuery(
          docs[0].pageContent,
        );
        console.log(
          `Embedding dimension: ${testEmbedding.length} (expected: 3072)`,
        );

        if (testEmbedding.length !== 3072) {
          console.warn(
            `Warning: Embedding dimension is ${testEmbedding.length}, expected 3072`,
          );
        }
      }

      for (let i = 0; i < totalBatches; i++) {
        const startIndex = i * batchSize;
        const endIndex = Math.min(startIndex + batchSize, docs.length);
        const batch = docs.slice(startIndex, endIndex);

        console.log(
          `Processing batch ${i + 1}/${totalBatches} (${batch.length} documents)`,
        );

        try {
          await this.vectorStore.addDocuments(batch);

          processedCount += batch.length;

          results.push({
            batchNumber: i + 1,
            documentsInBatch: batch.length,
            documentsProcessed: batch.length,
            startIndex,
            endIndex: endIndex - 1,
            status: 'success',
          });

          console.log(
            `✓ Batch ${i + 1} completed. Processed ${batch.length} documents. Total processed: ${processedCount}/${docs.length}`,
          );
        } catch (error: any) {
          console.error(`Error processing batch ${i + 1}:`, error);
          results.push({
            batchNumber: i + 1,
            documentsInBatch: batch.length,
            documentsProcessed: 0,
            startIndex,
            endIndex: endIndex - 1,
            status: 'error',
            error: error.message,
          });
        }

        if (i < totalBatches - 1) {
          await new Promise((resolve) => setTimeout(resolve, 2000));
        }
      }

      return {
        totalDocuments: docs.length,
        processedDocuments: processedCount,
        totalBatches,
        batchSize,
        embeddingModel: 'gemini-embedding-001',
        expectedEmbeddingDimension: 3072,
        results,
        summary: {
          successful: results.filter((r) => r.status === 'success').length,
          failed: results.filter((r) => r.status === 'error').length,
        },
      };
    } catch (error) {
      console.error('Error processing CSV:', error);
      throw new BadRequestException(`Failed to process CSV: ${error.message}`);
    } finally {
      if (tempFilePath) {
        try {
          await this.unlink(tempFilePath);
          console.log('Temporary file cleaned up');
        } catch (cleanupError) {
          console.error('Error cleaning up temporary file:', cleanupError);
        }
      }
    }
  }

  @Post('chat')
  async chat(
    @Body() request: { query: string; limit?: number; temperature?: number },
  ) {
    try {
      await this.ensureInitialized();
      const { query, limit = 4, temperature } = request;

      if (!query?.trim()) {
        throw new BadRequestException('Query is required');
      }

      if (temperature !== undefined) {
        this.model = new ChatGoogleGenerativeAI({
          model: 'gemini-2.0-flash',
          temperature: Math.max(0, Math.min(2, temperature)),
        });
      }

      const searchResults = await this.appService.performSimilaritySearch(
        query,
        limit,
        this.embeddings,
        this.vectorStore,
      );

      if (searchResults.length === 0) {
        return {
          response:
            "I couldn't find any relevant transaction data for your query.",
          query,
          resultsFound: 0,
        };
      }

      const response = await this.appService.generateResponse(
        query,
        searchResults,
        this.model,
      );

      return {
        response: response.content,
        query,
        resultsFound: searchResults.length,
        metadata: {
          topScores: searchResults.map((r: any) => r[1]),
          model: 'gemini-2.0-flash',
          embeddingModel: 'gemini-embedding-001',
        },
      };
    } catch (error: any) {
      console.error('Error in chat endpoint:', error);
      throw new BadRequestException(`Chat processing failed: ${error.message}`);
    }
  }

  @Get('summarize')
  async summarize() {
    await this.ensureInitialized();

    try {
      const transactionData = await this.appService.getTransactionData(
        this.embeddings,
        this.vectorStore,
      );

      if (!transactionData || transactionData.length === 0) {
        return {
          success: false,
          error: 'No transaction records found in the database',
          suggestion:
            'Please upload transaction data first using the /sync endpoint',
        };
      }

      const analysis = await this.appService.generateFinancialAnalysis(
        transactionData,
        this.model,
      );

      return {
        success: true,
        recordsAnalyzed: transactionData.length,
        period: this.appService.extractPeriodInfo(transactionData),
        insights: analysis,
        generatedAt: new Date().toISOString(),
      };
    } catch (error: any) {
      console.error('Financial summarization error:', error);
      return {
        success: false,
        error: 'Failed to generate financial summary',
        details: error.message,
        suggestion: 'Please check your transaction data format and try again',
      };
    }
  }
}
