import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { join } from 'path';
import { ConfigModule } from '@nestjs/config';
import { AppService } from './app.service';

@Module({
  imports: [
    ConfigModule.forRoot({
      envFilePath: join(__dirname, '../', '.env'),
      isGlobal: true,
    }),
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
