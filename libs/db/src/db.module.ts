import { Module, Global } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { NamingStrategy } from './naming.strategy';
import { DbService } from './db.service';
import { Device } from './device';
import { Internal } from './env';

@Global()
@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: 'localhost',
      port: 5432,
      username: 'postgres',
      password: '.fc12#$',
      database: 'farmconnect',
      entities: [__dirname + "/device/*.entity.ts", __dirname + "/env/*.entity.ts"],
      autoLoadEntities: true,
      namingStrategy: new NamingStrategy(),
      logging: true
    }),
    TypeOrmModule.forFeature([Device, Internal])
  ],
  controllers: [],
  providers: [DbService],
  exports: [TypeOrmModule, DbService],
})
export class DbModule {}
