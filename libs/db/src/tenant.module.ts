import { Global, Module } from "@nestjs/common";
import { TENANT_DATASOURCE } from "./constants";
import { DbModule } from "./db.module";
import { tenantFactory } from "./tenant.factory";

@Global()
@Module({
  imports: [DbModule],
  providers: [tenantFactory],
  exports: [TENANT_DATASOURCE]
})
export class TenantModule {}