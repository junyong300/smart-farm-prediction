import { ConfigService } from '@nestjs/config';
import { NestFactory } from '@nestjs/core';
import { Transport } from '@nestjs/microservices';
import { LogWrapper } from '@libs/config';
import { AppModule } from './app/app.module';

async function bootstrap() {
  const logWrapper = new LogWrapper("sensor");
  const app = await NestFactory.create(AppModule, {logger: logWrapper});
  const configService = app.get<ConfigService>(ConfigService);
  const logLevel = configService.get<string>('SENSOR_LOG_LEVEL', 'info');
  logWrapper.setLogLevel(logLevel);

  app.connectMicroservice({
    transport: Transport.REDIS,
    options: {
        url: 'redis://localhost:' + configService.get<number>('REDIS_PORT')
    }
  });

  await app.startAllMicroservices();
}

bootstrap();
