import { CommonConfigModule } from '@libs/config';
import { Module, Global } from '@nestjs/common';
import { COMMON } from './constants';
import { DbService } from './db.service';

@Global()
@Module({
  imports: [CommonConfigModule],
  providers: [
    {provide: COMMON, useValue: COMMON},
    DbService
  ],
  exports: [COMMON, DbService]
})
export class DbModule {}
