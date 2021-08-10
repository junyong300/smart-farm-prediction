import { Injectable } from '@nestjs/common';
import { Message } from '@smart-farm-prediction/api-interfaces';

@Injectable()
export class AppService {
  getData(): Message {
    return { message: 'Welcome to api!' };
  }
}
