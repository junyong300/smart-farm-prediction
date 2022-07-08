import { Column, Entity, PrimaryGeneratedColumn } from 'typeorm';

@Entity({ name: 'forward' })
export class Forward {
  @PrimaryGeneratedColumn({ comment: 'serial' })
  id: number;
  @Column({ comment: 'host' })
  host: string
  @Column({ comment: 'comma seperated serial. regular ex.' })
  serials: string;
  @Column({ comment: 'enabled' })
  enabled: boolean;
  @Column()
  description: string;
  @Column({ name: "create_dt" })
  createDt: Date;
  @Column({ name: "modify_dt" })
  modifyDt: Date;
}
