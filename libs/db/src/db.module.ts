import { Module, Global } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { DbService } from './db.service';
import { Internal } from './sensor/internal.entity';

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
      entities: [__dirname + "/sensor/*.entity.ts"],
      autoLoadEntities: true,
    }),
    Internal
  ],
  controllers: [],
  providers: [DbService],
  exports: [Internal, DbService],
})
export class DbModule {}
