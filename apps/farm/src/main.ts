import { NestFactory } from '@nestjs/core';
import { ConfigService } from '@nestjs/config';
import { Transport } from '@nestjs/microservices';
import { LogWrapper } from '@libs/config';
import { AppModule } from './app/app.module';

async function bootstrap() {
  const logWrapper = new LogWrapper("farm");
  const app = await NestFactory.create(AppModule, {logger: logWrapper});
  const configService = app.get<ConfigService>(ConfigService);
  const logLevel = configService.get<string>('FARM_LOG_LEVEL', 'info');
  logWrapper.setLogLevel(logLevel);

  app.connectMicroservice({
    transport: Transport.REDIS,
    options: {
        url: 'redis://localhost:' + configService.get<number>('REDIS_PORT')
        //url: 'redis://' + configService.get<string>('REDIS_HOST', 'localhost') + ':' + configService.get<number>('REDIS_PORT', 6379)
    }
  });

  await app.startAllMicroservices();
}

bootstrap();
