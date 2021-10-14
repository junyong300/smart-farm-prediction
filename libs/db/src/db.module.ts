import { Module, Global } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { CommonConfigModule } from '@lib/config';
import { ConfigService } from '@nestjs/config';
import { NamingStrategy } from './naming.strategy';
import { DbService } from './db.service';
import { Device } from './device';
import { Internal } from './env';

@Global()
@Module({
  imports: [
    CommonConfigModule,
    TypeOrmModule.forRootAsync({
      useFactory: (config: ConfigService) => ({
        type: config.get<"postgres"|"mysql">('DB_TYPE'),
        host: config.get<string>('DB_HOST'),
        port: config.get<number>('DB_PORT'),
        username: config.get<string>('DB_USER'),
        password: config.get<string>('DB_PASSWORD'),
        database: config.get<string>('DB_DATABASE'),
        entities: [__dirname + "/device/*.entity.ts", __dirname + "/env/*.entity.ts"],
        autoLoadEntities: true,
        namingStrategy: new NamingStrategy(),
        logging: true
      }),
      inject: [ConfigService],
    }),
    TypeOrmModule.forFeature([Device, Internal])
  ],
  controllers: [],
  providers: [DbService],
  exports: [TypeOrmModule, DbService],
})
export class DbModule {}
