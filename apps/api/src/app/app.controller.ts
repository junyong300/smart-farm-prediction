import { Body, Controller, Get, Inject, Param, Post, Query } from '@nestjs/common';
import { ClientProxy } from '@nestjs/microservices'
import { timeout } from 'rxjs/operators';
import { AppService } from './app.service';

@Controller()
export class AppController {
  constructor(@Inject('REDIS') private redis: ClientProxy) {}

  @Get('/api/cmd.json')
  getCmd(@Query() q) {
    console.log(q);
    return this.redis.send(q['cmd'], q['data']).pipe(timeout(3000));
  }

  @Post('/api/cmd.json')
  postCmd(@Body() body) {
    console.log(body);
    return this.redis.send(body['cmd'], body['data']).pipe(timeout(3000));
  }
}
