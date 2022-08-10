import { CommonConfigService } from "@libs/config";
import { Injectable } from "@nestjs/common";
import { DataSource, EntityTarget, ObjectLiteral } from "typeorm";
import { Forward } from "./common";
import { Device, SdhCultureMedium, SdhExternal, SdhInternal } from "./sensor";
import { typeormFactory } from "./typeorm-options";

@Injectable()
export class DbService {
  private dataSources = new Map<string, DataSource>();
  // TODO: common과 나머지를 나눌까?
  private entities = [
    Forward, SdhInternal, SdhExternal, SdhCultureMedium, Device
  ];

  constructor(private config: CommonConfigService) {}

  /**
   * get or create dataSource
   * @param name database or database token
   * @param entities
   * @returns 
   */
  async getDataSource(name?: string, entities?) {
    let ds = this.dataSources.get(name || 'default');
    if (!ds) {
      ds = new DataSource({...typeormFactory(this.config, name), entities: entities || this.entities});
      this.dataSources.set(name || "default", ds);
    }
    if (!ds.isInitialized) {
      await ds.initialize();
    }
    
    return ds;
  }

  getCustomRepository<Entity extends ObjectLiteral>(dataSource: DataSource, target: EntityTarget<Entity>) {
    return dataSource.getRepository(target).extend({});
  }
}

