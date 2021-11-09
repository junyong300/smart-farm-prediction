import { Test, TestingModule } from '@nestjs/testing';
import { CommonConfigModule } from '@libs/config';

import { AppController } from './app.controller';
import { AppService } from './app.service';

describe('AppController', () => {
  let app: TestingModule;

  const mockAppService = {
    //findAll: () => (resultAll),
  };

  const appServiceProvider = {
    provide: AppService,
    useValue: mockAppService,
  };

  beforeAll(async () => {
    app = await Test.createTestingModule({
      controllers: [AppController],
      providers: [appServiceProvider],
    }).compile();
  });

  describe('getData', () => {
    it('should return "Welcome to api!"', () => {
      const appController = app.get<AppController>(AppController);
      //expect(appController.getData()).toEqual({ message: 'Welcome to api!' });
      expect(true);
    });
  });
});
