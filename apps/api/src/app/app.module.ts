import { Module, NestModule, MiddlewareConsumer, RequestMethod } from '@nestjs/common';
import { ClientProxyFactory, Transport } from '@nestjs/microservices';
// import { ServeStaticModule } from '@nestjs/serve-static';
import { CommonConfigModule } from '@libs/config';
import { AppService } from './app.service';
import { AppController } from './app.controller';
import { SensorController } from './controllers/sensor.controller';
import { ConfigService } from '@nestjs/config';
import { FrontendMiddleware } from './middlewares/frontend.middleware';
import { HttpModule } from '@nestjs/axios';
//import { join } from 'path';

@Module({
  imports: [
    /*
    ServeStaticModule.forRoot({
      rootPath: join(__dirname, 'assets')
    }),
    */
    HttpModule,
    CommonConfigModule,
    ConfigService
  ],
  controllers: [AppController, SensorController],
  providers: [AppService,
    {
      provide: 'REDIS',
      useFactory: (configService: ConfigService) => {
        return ClientProxyFactory.create({
          transport: Transport.REDIS,
          options: {
            url: 'redis://localhost:' + configService.get<number>('REDIS_PORT'),
          }
        });
      },
      inject: [ConfigService]
    }
  ]
})
export class AppModule implements NestModule {
  configure(consumer: MiddlewareConsumer): void {
    consumer.apply(FrontendMiddleware).forRoutes(
      {
        path: '/**', // For all routes
        method: RequestMethod.ALL, // For all methods
      },
    );
  }

}
