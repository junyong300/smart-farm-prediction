import { BadRequestException, Scope } from "@nestjs/common";
import { REQUEST } from "@nestjs/core";
import { EntitiesMetadataStorage } from "@nestjs/typeorm/dist/entities-metadata.storage";
import { COMMON, TENANT, TENANT_DATASOURCE } from "./constants";
import { Tenant } from "./common";
import { DbService } from "./db.service";
import { RpcException } from "@nestjs/microservices";

export const tenantFactory = {
  provide: TENANT_DATASOURCE,
  scope: Scope.REQUEST,
  useFactory: async (payload, dbService: DbService) => {
    const siteCode = payload?.data?.siteCode;
    let database;

    // get tenant database
    if (siteCode) {
      const commonDs = await dbService.getDataSource(COMMON);
      const tenantRepo = commonDs.getRepository(Tenant);
      const tenant = await tenantRepo.findOne({where: {code: siteCode}, cache: 60 * 5 * 1000}); // 5분
      database = tenant?.dbName;
      if (!database) {
        throw new RpcException(new BadRequestException('No siteCode'));
      }
    } else {
      // siteCode가 없을 경우 어떻게 처리할 것인지 결정해야 한다.
      // 현재는 default값으로 처리하고 있다
    }

    const entities = EntitiesMetadataStorage.getEntitiesByDataSource(TENANT); // TypeormModule의 autoLoadEntities 옵션 구현
    return dbService.getDataSource(database, entities);
  },
  inject: [REQUEST, DbService]
}