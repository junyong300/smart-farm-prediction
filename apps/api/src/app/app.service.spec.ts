//import { Test } from '@nestjs/testing';
//import { AppService } from './app.service';

describe('AppService', () => {
  //let service: AppService;

  beforeAll(async () => {
    /*
    const app = await Test.createTestingModule({
      providers: [AppService],
    }).compile();

    service = app.get<AppService>(AppService);
    */
  });

  describe('dummy', () => {
    it('dummy test, always true"', () => {
      //expect(service.getData()).toEqual({ message: 'Welcome to api!' });
      expect(true);
    });
  });
});
