import { Module } from '@nestjs/common';
import { DbModule } from '@lib/db';

import { AppController } from './app.controller';
import { AppService } from './app.service';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Internal } from '@lib/db/sensor';

@Module({
  imports: [
    DbModule, 
    TypeOrmModule.forFeature([Internal]),
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
