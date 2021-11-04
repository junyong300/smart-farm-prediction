import { Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { NestFactory } from '@nestjs/core';
//import { WsAdapter } from '@nestjs/platform-ws';
import { SwaggerModule, DocumentBuilder} from '@nestjs/swagger';
import { LogWrapper } from '@lib/config';

import { AppModule } from './app/app.module';

async function bootstrap() {
  const logWrapper = new LogWrapper("api");
  const app = await NestFactory.create(AppModule, { logger: logWrapper });
  const configService = app.get(ConfigService);
  const logLevel = configService.get<string>('API_LOG_LEVEL', 'info');
  logWrapper.setLogLevel(logLevel);

  const swaggerConfig = new DocumentBuilder()
    .setTitle('Edge api')
    .setDescription("API Description")
    .setVersion('1.0')
    .build();
  const document = SwaggerModule.createDocument(app, swaggerConfig);
  SwaggerModule.setup('swagger', app, document);

  app.enableCors({ origin: '*' });
  // app.useWebSocketAdapter(new WsAdapter(app));
  const port = process.env.PORT || configService.get('API_PORT') || 8080;
  await app.listen(port, () => {
    Logger.log('Listening at http://0.0.0.0:' + port);
  });
}

bootstrap();
