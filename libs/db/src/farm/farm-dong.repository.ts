import { DataSource } from "typeorm";
import { FarmDong } from "./farm-dong.entity";

export const FarmDongRepository = (dataSource: DataSource) => 
    dataSource.getRepository(FarmDong).extend({
      async getDeviceIdByDongId(dongId: number) {
        const rs = await this.createQueryBuilder()
            .select("in_env_device_idx")
            .where("farm_dong_idx = :dongId", {dongId: dongId})
            .getRawOne();
        
        return rs['in_env_device_idx']
      }
    });

/* @EntityRepository(FarmDong)
export class FarmDongRepository extends Repository<FarmDong> {
   
  async getDeviceIdByDongId(dongId: number) {
    const rs = await this.createQueryBuilder()
        .select("in_env_device_idx")
        .where("farm_dong_idx = :dongId", {dongId: dongId})
        .getRawOne();
    
    return rs['in_env_device_idx']
  }
} */