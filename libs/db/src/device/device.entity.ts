import { Entity, Column, BaseEntity, PrimaryColumn } from "typeorm";

/**
 * 내부환경센서
 */
@Entity({ schema: 'device'})
export class Device extends BaseEntity {
    @PrimaryColumn()
    id: number;
    @Column()
    farmId: number;
    @Column()
    dongId: number;
    @Column()
    commType: number;
    @Column()
    modelId: number;
    @Column()
    serial: string;
    @Column()
    formatId: number;
    @Column()
    sensingInterval: number;
    @Column()
    state: number;
    
    @Column()
    firstActiveTime: Date; 
    @Column()
    lastActiveTime: Date; 

    @Column()
    createdTime: Date; 
    @Column()
    modifiedTime: Date; 
    @Column()
    deletedTime: Date; 
}