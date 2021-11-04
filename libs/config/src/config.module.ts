import { Module, Global } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import * as path from 'path';

const MODE = process.env.NODE_ENV;
const DEV_CONFIG_FILE = process.argv.slice(2)[0];

@Global()
@Module({
  imports: [
    ConfigModule.forRoot({
      // use args or use different conf file by NODE_ENV
      envFilePath: [DEV_CONFIG_FILE ? DEV_CONFIG_FILE :
        (MODE && MODE.startsWith('dev')) ? 'resources/config/common.conf' : path.join(__dirname, '..', 'config/common.conf'),
        path.join(__dirname, '..', 'config/common.conf')
      ]
    })
  ],
  controllers: [],
  providers: [ConfigService],
  exports: [ConfigService],
})
export class CommonConfigModule {}
