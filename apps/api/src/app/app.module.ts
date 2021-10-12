import { Module } from '@nestjs/common';
import { ClientProxyFactory, ClientsModule, Transport } from '@nestjs/microservices';
import { ServeStaticModule } from '@nestjs/serve-static';
import { CommonConfigModule } from '@lib/config';
import { AppService } from './app.service';
import { AppController } from './app.controller';
import { SensorController } from './controllers/sensor.controller';
import { join } from 'path';
import { ConfigService } from '@nestjs/config';

@Module({
  imports: [
    ServeStaticModule.forRoot({
      rootPath: join(__dirname, 'assets')
    }),
    CommonConfigModule
  ],
  controllers: [AppController, SensorController],
  providers: [AppService,
    {
      provide: 'REDIS',
      useFactory: (configService: ConfigService) => {
        return ClientProxyFactory.create({
          transport: Transport.REDIS,
          options: {
            url: 'redis://localhost:' + configService.get<number>('REDIST_PORT'),
          }
        });
      },
      inject: [ConfigService]
    }
  ]
})
export class AppModule {}
