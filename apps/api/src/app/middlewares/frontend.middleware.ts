import { Injectable, NestMiddleware } from '@nestjs/common';
import * as path from 'path';

const allowedExt = [
  '.js',
  '.ico',
  '.css',
  '.png',
  '.jpg',
  '.woff2',
  '.woff',
  '.ttf',
  '.svg',
];

const resolvePath = (file: string) => path.resolve(path.join(__dirname, '..', `frontend/edge-mon/${file}`));

@Injectable()
export class FrontendMiddleware implements NestMiddleware {
  use(req, res, next: () => void) {
    const { url } = req;
      // it starts with /api or /connfarm --> continue with execution
    if (url.indexOf('api') === 1 || url.indexOf('connfarm') === 1) {
      next();
    } else if (allowedExt.filter(ext => url.indexOf(ext) > 0).length > 0) {
      // it has a file extension --> resolve the file
      res.sendFile( resolvePath(url));
    } else {
      // in all other cases, redirect to the index.html!
      res.sendFile(resolvePath('index.html'));
    }
  }
}
