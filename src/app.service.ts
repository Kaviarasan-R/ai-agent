import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableLambda, RunnableSequence } from '@langchain/core/runnables';
import { Injectable } from '@nestjs/common';

@Injectable()
export class AppService {
  constructor() {}

  /* Chat */

  async performSimilaritySearch(
    query: string,
    limit: number,
    embeddings: any,
    vectorStore: any,
  ) {
    try {
      const queryVector = await embeddings.embedQuery(query);

      const results = await vectorStore.similaritySearchVectorWithScore(
        queryVector,
        limit,
      );

      const relevanceThreshold = 0.5;
      return results.filter(([, score]) => score >= relevanceThreshold);
    } catch (error) {
      console.error('Error in similarity search:', error);
      throw error;
    }
  }

  async generateResponse(query: string, searchResults: any[], model: any) {
    try {
      const systemPrompt = this.getSystemPrompt(query);

      const promptTemplate = ChatPromptTemplate.fromMessages([
        ['system', systemPrompt],
        ['user', 'Context: {context}\n\nQuery: {query}'],
      ]);

      const formattedContext = searchResults
        .map((result, index) => {
          const [doc, score] = result;
          return `Record ${index + 1} (Relevance: ${(score * 100).toFixed(1)}%):\n${doc.pageContent}`;
        })
        .join('\n\n---\n\n');

      const contextFormatter = RunnableLambda.from(() => ({
        context: formattedContext,
        query,
      }));

      const ragChain = RunnableSequence.from([
        contextFormatter,
        promptTemplate,
        model,
      ]);

      return await ragChain.invoke({ query });
    } catch (error) {
      console.error('Error generating response:', error);
      throw error;
    }
  }

  getSystemPrompt(query: string): string {
    const lowerQuery = query.toLowerCase();

    if (
      lowerQuery.includes('maximum') ||
      lowerQuery.includes('highest') ||
      lowerQuery.includes('largest')
    ) {
      return `You are a financial data analyst. From the provided transaction records, identify and present the highest value transactions. 
    Format your response as: "Amount: $X.XX | Type: [transaction_type] | Additional details if relevant"
    Focus on the transactions with the highest amounts and provide clear, actionable insights.`;
    }

    if (
      lowerQuery.includes('minimum') ||
      lowerQuery.includes('lowest') ||
      lowerQuery.includes('smallest')
    ) {
      return `You are a financial data analyst. From the provided transaction records, identify and present the lowest value transactions.
    Format your response clearly with amount and transaction type.`;
    }

    if (lowerQuery.includes('summary') || lowerQuery.includes('overview')) {
      return `You are a financial data analyst. Provide a comprehensive summary of the transaction data.
    Include key metrics like total amounts, transaction types, and notable patterns.`;
    }

    if (lowerQuery.includes('pattern') || lowerQuery.includes('trend')) {
      return `You are a financial data analyst. Analyze the transaction data for patterns and trends.
    Highlight recurring transaction types, amounts, and any notable behaviors.`;
    }

    return `You are a helpful financial data analyst. Analyze the provided transaction records and answer the user's query clearly and concisely.
  Present the information in an organized format with amounts, transaction types, and relevant details.`;
  }

  /* Summarization */

  async getTransactionData(embeddings: any, vectorStore: any) {
    const vector = await embeddings.embedQuery(
      'financial transaction bank payment deposit withdrawal expense income',
    );

    const results = await vectorStore.similaritySearchVectorWithScore(
      vector,
      50,
    );

    return results.filter(([, score]) => score >= 0.6);
  }

  async generateFinancialAnalysis(transactionData: any[], model: any) {
    const promptTemplate = ChatPromptTemplate.fromMessages([
      [
        'system',
        `You are an expert financial analyst. Analyze the transaction records and provide comprehensive insights.
      
      IMPORTANT: Return ONLY a valid JSON object with this EXACT structure:
      {{
        "totalTransactions": {{
          "count": <number>,
          "description": "Total number of transactions processed in the selected period"
        }},
        "totalAmountInOut": {{
          "totalInflow": <number>,
          "totalOutflow": <number>,
          "netAmount": <number>,
          "description": "Total money received vs spent, with net position"
        }},
        "bankwiseSummary": {{
          "banks": [
            {{
              "bankName": "<string>",
              "inflow": <number>,
              "outflow": <number>,
              "netPosition": <number>
            }}
          ],
          "description": "Bank-wise breakdown of inflow and outflow amounts"
        }},
        "top5ContactsByExpense": {{
          "contacts": [
            {{
              "contactName": "<string>",
              "totalExpense": <number>,
              "transactionCount": <number>
            }}
          ],
          "description": "Highest paid vendors, contacts, or payees ranked by expense amount"
        }},
        "accountwiseSpending": {{
          "accounts": [
            {{
              "accountType": "<string>",
              "accountNumber": "<string>",
              "totalSpent": <number>,
              "transactionCount": <number>
            }}
          ],
          "description": "Spending breakdown by different accounts"
        }},
        "monthlyTrend": {{
          "months": [
            {{
              "month": "<YYYY-MM>",
              "inflow": <number>,
              "outflow": <number>,
              "netAmount": <number>,
              "transactionCount": <number>
            }}
          ],
          "description": "Monthly transaction trends showing spending and income patterns"
        }},
        "transactionTypeDistribution": {{
          "types": [
            {{
              "type": "<string>",
              "count": <number>,
              "totalAmount": <number>,
              "percentage": <number>
            }}
          ],
          "description": "Distribution of transaction types (payment, deposit, withdrawal, etc.)"
        }},
        "keyInsights": {{
          "highestExpenseMonth": "<string>",
          "lowestExpenseMonth": "<string>",
          "averageMonthlySpending": <number>,
          "mostFrequentTransactionType": "<string>",
          "largestSingleTransaction": <number>,
          "financialHealthScore": <number>,
          "recommendations": ["<string>", "<string>", "<string>"]
        }},
        "executiveSummary": "<comprehensive 2-3 sentence summary of financial position>"
      }}

      ANALYSIS INSTRUCTIONS:
      - Extract amounts by looking for currency symbols (₹, $, €), numbers with decimals
      - Identify transaction types: payment, deposit, withdrawal, transfer, salary, investment, etc.
      - Parse dates to determine monthly trends
      - Identify bank names, account numbers, and contact/vendor names
      - Calculate percentages for distribution analysis
      - Provide actionable financial insights and recommendations
      - Financial health score should be 1-100 based on spending patterns, savings rate, etc.
      
      Do not include any text outside the JSON object.`,
      ],
      [
        'user',
        'Analyze these transaction records and provide comprehensive financial insights:\n\n{context}',
      ],
    ]);

    const contextProcessor = RunnableLambda.from(() => {
      const context = transactionData
        .map((item: any, index: number) => {
          const content = item[0].pageContent;
          const relevanceScore = item[1];
          return `Record ${index + 1} (Relevance: ${(relevanceScore * 100).toFixed(1)}%):\n${content}`;
        })
        .join('\n\n' + '='.repeat(50) + '\n\n');

      return { context };
    });

    const enhancedJsonParser = RunnableLambda.from((output: any) => {
      try {
        const content = output.content || output.toString();
        let jsonStr = content.trim();

        jsonStr = jsonStr.replace(/```json\s*|\s*```/g, '');
        jsonStr = jsonStr.replace(/^[^{]*({[\s\S]*})[^}]*$/, '$1');

        const parsed = JSON.parse(jsonStr);

        const requiredSections = [
          'totalTransactions',
          'totalAmountInOut',
          'bankwiseSummary',
          'top5ContactsByExpense',
          'accountwiseSpending',
          'monthlyTrend',
          'transactionTypeDistribution',
          'keyInsights',
        ];

        for (const section of requiredSections) {
          if (!(section in parsed)) {
            console.warn(`Missing section: ${section}`);
          }
        }

        return parsed;
      } catch (error: any) {
        console.error('Enhanced JSON parsing error:', error);
        console.error('Raw AI output:', output.content || output);

        return this.generateFallbackAnalysis(transactionData, output);
      }
    });

    const analysisChain = RunnableSequence.from([
      contextProcessor,
      promptTemplate,
      model,
      enhancedJsonParser,
    ]);

    return await analysisChain.invoke({});
  }

  generateFallbackAnalysis(transactionData: any[], rawOutput: any) {
    return {
      totalTransactions: {
        count: transactionData.length,
        description:
          'Total number of transactions processed in the selected period',
      },
      totalAmountInOut: {
        totalInflow: 0,
        totalOutflow: 0,
        netAmount: 0,
        description:
          'Unable to calculate due to parsing error - please check data format',
      },
      bankwiseSummary: {
        banks: [],
        description: 'Bank analysis unavailable due to processing error',
      },
      top5ContactsByExpense: {
        contacts: [],
        description: 'Contact analysis unavailable due to processing error',
      },
      accountwiseSpending: {
        accounts: [],
        description: 'Account analysis unavailable due to processing error',
      },
      monthlyTrend: {
        months: [],
        description:
          'Monthly trend analysis unavailable due to processing error',
      },
      transactionTypeDistribution: {
        types: [],
        description:
          'Transaction type analysis unavailable due to processing error',
      },
      keyInsights: {
        highestExpenseMonth: 'Unknown',
        lowestExpenseMonth: 'Unknown',
        averageMonthlySpending: 0,
        mostFrequentTransactionType: 'Unknown',
        largestSingleTransaction: 0,
        financialHealthScore: 50,
        recommendations: [
          'Please ensure transaction data is properly formatted',
          'Check if amount and date fields are clearly specified',
          'Verify bank and account information is included in records',
        ],
      },
      executiveSummary: `Analysis of ${transactionData.length} transaction records encountered processing errors. Please review data format and try again.`,
      processingError: true,
      rawResponse: rawOutput.content || rawOutput.toString(),
    };
  }

  extractPeriodInfo(transactionData: any[]) {
    const dates = transactionData
      .map((item) => {
        const content = item[0].pageContent;
        // Simple date extraction - you might want to improve this based on your data format
        const dateMatch = content.match(
          /\d{4}-\d{2}-\d{2}|\d{2}\/\d{2}\/\d{4}|\d{2}-\d{2}-\d{4}/,
        );
        return dateMatch ? dateMatch[0] : null;
      })
      .filter(Boolean);

    if (dates.length === 0) {
      return {
        startDate: 'Unknown',
        endDate: 'Unknown',
        description: 'Transaction period could not be determined',
      };
    }

    const sortedDates = dates.sort();
    return {
      startDate: sortedDates[0],
      endDate: sortedDates[sortedDates.length - 1],
      description: `Analysis covers transactions from ${sortedDates[0]} to ${sortedDates[sortedDates.length - 1]}`,
    };
  }
}
