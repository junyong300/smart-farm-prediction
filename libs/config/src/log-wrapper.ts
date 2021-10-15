/* eslint-disable @typescript-eslint/no-unused-vars */
import { LoggerService, LogLevel } from "@nestjs/common";
import { configure, getLogger } from "log4js";
import * as path from 'path';

export class LogWrapper implements LoggerService {
  private readonly logger = getLogger();

  constructor(moduleName: string, level: string) {
    this.logger.setParseCallStackFunction(this.parseCallStack);
    this.setLog4jOption(moduleName, level);
  }

  log(message: any, ...optionalParams: any[]) {
    this.logger.info(message);
  }
  error(message: any, ...optionalParams: any[]) {
    this.logger.error(message);
  }
  warn(message: any, ...optionalParams: any[]) {
    this.logger.warn(message);
  }
  debug?(message: any, ...optionalParams: any[]) {
    this.logger.debug(message);
  }
  verbose?(message: any, ...optionalParams: any[]) {
    this.logger.trace(message);
  }
  setLogLevels?(levels: LogLevel[]) {
    return;
  }

  setLog4jOption(moduleName: string, level: string) {
    const fileName = path.join(__dirname, "..", "logs", moduleName + ".log");
    const filePattern = '%d{yyyy-MM-dd hh:mm:ss.SSS} %-5p [%f{2}:%l] %m';
    const outPattern = '%d{yyyy-MM-dd hh:mm:ss.SSS} %[%-5p%] [%f{2}:%l] %m';
    configure({
      appenders: {
        file: { type: "file", filename: fileName, maxLogSize: 104857600, backups: 7, keepFileExt: true, 
          layout: { type: 'pattern', pattern: filePattern }},
        stdout: { type: "stdout", layout: { type: 'pattern', pattern: outPattern }}
      },
      categories: { default: { appenders: ["stdout", "file"], level: level, enableCallStack: true }}
    });
  }

  parseCallStack(data, skipIdx = 7) {
    const stackReg = /at (?:(.+)\s+\()?(?:(.+?):(\d+)(?::(\d+))?|([^)]+))\)?/;
    const stacklines = data.stack.split("\n").slice(skipIdx);
    const lineMatch = stackReg.exec(stacklines[0]);
    if (lineMatch && lineMatch.length === 6) {
      return {
        functionName: lineMatch[1],
        fileName: lineMatch[2],
        lineNumber: parseInt(lineMatch[3], 10),
        columnNumber: parseInt(lineMatch[4], 10),
        callStack: stacklines.join("\n")
      };
    }
    return null;
  }
}