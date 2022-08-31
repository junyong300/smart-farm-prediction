import { Catch, ExceptionFilter } from '@nestjs/common';
import { Observable, throwError } from 'rxjs';
import { RpcException } from '@nestjs/microservices';

/**
 * Exception을 RpcException으로 변환
 */
@Catch()
export class CommExceptionFilter implements ExceptionFilter {
  catch(exception: Error): Observable<any> {
    return throwError(() => new RpcException(exception));
  }
}