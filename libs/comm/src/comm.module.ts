import { CommonConfigModule } from '@libs/config';
import { Module, Global } from '@nestjs/common';
import { CommOptionsService } from './comm-options.service';
import { CommService } from './comm.service';

@Global()
@Module({
  imports: [CommonConfigModule],
  providers: [CommOptionsService],
  exports: [CommonConfigModule, CommOptionsService],
})
export class CommModule {
  static forSend() {
    return {
      module: CommModule,
      providers: [CommService,
        {
          provide: 'REDIS',
          useFactory: (commOptionsService: CommOptionsService) => { return commOptionsService.getClientProxy(); },
          inject: [CommOptionsService],
        },
      ],
      exports: [CommService, 'REDIS']
    };
  }
}
