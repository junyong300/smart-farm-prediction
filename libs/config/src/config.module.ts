import { Module, Global } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import * as path from 'path';

const MODE = process.env.NODE_ENV;
const PARAM_CONFIG_FILE = process.argv.slice(2)[0];
const midDir = MODE?.startsWith('dev') ? '/devenv': '';
const configFile = PARAM_CONFIG_FILE ? PARAM_CONFIG_FILE :
  __dirname.includes('/dist/') ?  path.join(__dirname, "..", "..", "devenv", 'config', 'common.conf') :
  path.join(__dirname, "..", 'config', 'common.conf');
console.log("configFile: " + configFile);

@Global()
@Module({
  imports: [
    ConfigModule.forRoot({
      // use args or use different conf file by NODE_ENV
      envFilePath: [configFile]
    })
  ],
  controllers: [],
  providers: [ConfigService],
  exports: [ConfigService],
})
export class CommonConfigModule {}
