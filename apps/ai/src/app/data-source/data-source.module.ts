import { SdhCultureMedium, SdhExternal, SdhInternal } from '@libs/db/sensor';
import { Global, Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { DbService } from './db.service';

@Global()
@Module({
  imports: [
    TypeOrmModule.forFeature([SdhInternal, SdhExternal, SdhCultureMedium]),
  ],
  providers: [DbService],
  exports: [DbService]
})
export class DataSourceModule {}
