import { CommonConfigService } from "@libs/config";
import { TypeOrmModuleAsyncOptions } from "@nestjs/typeorm";
import { COMMON } from "./constants";

export const typeormFactory = (config: CommonConfigService, name?: string) => {
  let dbConfig = config.db;
  let database = name;

  switch (name) {
    case COMMON: database = 'common'; break;
  }

  return {
    name: name || "default",
    type: dbConfig.type,
    host: dbConfig.host,
    port: dbConfig.port,
    username: dbConfig.username,
    password: dbConfig.password,
    database: database || dbConfig.database,
    autoLoadEntities: true,
    keepConnectionAlive: true,
    logging: config.typeormLogEnabled,
    cache: config.getTypeOrmCacheOption() // redis

    //entities: [__dirname + '/**/*.entity{.ts,.js}'],
  };
};

export const typeormOptions: TypeOrmModuleAsyncOptions = {
  useFactory: typeormFactory,
  inject: [CommonConfigService]
};

export const typeormCommonOptions: TypeOrmModuleAsyncOptions = {
  name: COMMON,
  useFactory: typeormFactory,
  inject: [CommonConfigService, {token: COMMON, optional: true}]
};
