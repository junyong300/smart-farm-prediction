import { Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { NestFactory } from '@nestjs/core';
import { Transport } from '@nestjs/microservices';
import { AppModule } from './app/app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, {});
  const configService = app.get<ConfigService>(ConfigService);

  app.connectMicroservice({
    transport: Transport.REDIS,
    options: {
        url: 'redis://localhost:' + configService.get<number>('REDIS_PORT')
    }
  });

  await app.startAllMicroservices();
}

bootstrap();
