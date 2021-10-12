import { Injectable } from '@nestjs/common';
import { Message } from '@lib/api-interfaces';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class AppService {
  constructor(private config: ConfigService) {
  }
  getData(): Message {
    // return { message: 'Welcome to api!' };
    console.log(this.config.get<string>('DB_HOST'));
    return { message: this.config.get<string>('DB_HOST') };
  }
}
