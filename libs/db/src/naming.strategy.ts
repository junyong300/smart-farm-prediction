import { DefaultNamingStrategy, NamingStrategyInterface } from 'typeorm';
//import { snakeCase } from 'typeorm/util/StringUtils';

export class NamingStrategy extends DefaultNamingStrategy implements NamingStrategyInterface {
  tableName(className: string, customName: string): string {
    return customName ? lowerCase(customName) : lowerCase(className);
  }

  columnName(propertyName: string, customName: string, embeddedPrefixes: string[]): string {
    return (lowerCase(embeddedPrefixes.concat('').join('_')) + (customName ? lowerCase(customName) : lowerCase(propertyName)));
  }

  relationName(propertyName: string): string {
    return lowerCase(propertyName);
  }

  joinColumnName(relationName: string, referencedColumnName: string): string {
    return lowerCase(relationName + '_' + referencedColumnName);
  }

  joinTableName(firstTableName: string, secondTableName: string, firstPropertyName: string, secondPropertyName: string): string {
    return lowerCase(firstTableName + '_' + firstPropertyName.replace(/\./gi, '_') + '_' + secondTableName);
  }

  joinTableColumnName(tableName: string, propertyName: string, columnName?: string): string {
    return lowerCase(tableName + '_' + (columnName ? columnName : propertyName),
    );
  }

  classTableInheritanceParentColumnName(parentTableName: any, parentTableIdPropertyName: any): string {
    return lowerCase(parentTableName + '_' + parentTableIdPropertyName);
  }

  eagerJoinRelationAlias(alias: string, propertyPath: string): string {
    return lowerCase(alias + '__' + propertyPath.replace('.', '_'));
  }

}

function lowerCase(name: string) {
  return name.toLowerCase();
}