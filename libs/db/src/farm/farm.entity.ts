import { Entity, Column, PrimaryColumn } from "typeorm";

@Entity({ schema: 'sensor'})
export class Farm {
    @PrimaryColumn()
    idx!: number;
    @Column()
    temp: number;
}