import { Controller, Get, Inject } from '@nestjs/common';
import { ClientProxy } from '@nestjs/microservices'
import { timeout } from 'rxjs/operators';
import { Message } from '@lib/api-interfaces';
import { AppService } from './app.service';

@Controller()
export class AppController {
  constructor(private readonly appService: AppService, @Inject('REDIS') private redis: ClientProxy) {}

  @Get('hello')
  getData(): Message {
    return this.appService.getData();
  }

}
