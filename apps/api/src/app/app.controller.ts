import { Body, Controller, Get, Inject, Post, Query, Res } from '@nestjs/common';
import { ClientProxy } from '@nestjs/microservices'
import { timeout } from 'rxjs/operators';
import { Logger } from "@nestjs/common";

//import { AppService } from './app.service';
import * as path from 'path';
import { ApiQuery } from '@nestjs/swagger';

@Controller()
export class AppController {
  //private readonly logger = getLogger();
  private readonly logger = new Logger(AppController.name);
  constructor(@Inject('REDIS') private redis: ClientProxy) {}

  @Get('')
  root(@Res() response): void {
    // the homepage will load our index.html which contains angular logic
    response.sendFile(path.resolve(path.join(__dirname, '..', 'frontend/edge-mon/index.html')));
  }

  @ApiQuery({name: 'data', required: true})
  @ApiQuery({name: 'cmd', required: true})
  @Get('/api/cmd.json')
  getCmd(@Query() q) {
    this.logger.debug(q);
    return this.redis.send(q['cmd'], q['data']).pipe(timeout(5000));
  }

  @Post('/api/cmd.json')
  postCmd(@Body() body) {
    return this.redis.send(body['cmd'], body['data']).pipe(timeout(5000));
  }
}
