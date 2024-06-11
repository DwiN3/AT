grammar CSharp;

compilationUnit : (usingDirective | globalAttribute | namespaceDeclaration | typeDeclaration)* EOF ;

usingDirective : 'using' (IDENTIFIER ('.' IDENTIFIER)* ';' | 'static' 'void' IDENTIFIER '(' parameterList? ')' block) ;

globalAttribute : '[' 'assembly' ':' attributeList ']' ;

namespaceDeclaration : 'namespace' qualifiedIdentifier '{' (usingDirective | typeDeclaration)* '}' ;

typeDeclaration : classDeclaration
                | interfaceDeclaration
                | structDeclaration
                | enumDeclaration ;

classDeclaration : 'partial'? 'class' IDENTIFIER ('<' typeParameterList '>')? ('base' baseList)? '{' classBody '}' ;

interfaceDeclaration : 'interface' IDENTIFIER ('<' typeParameterList '>')? ('base' baseList)? '{' interfaceBody '}' ;

structDeclaration : 'struct' IDENTIFIER ('<' typeParameterList '>')? ('base' baseList)? '{' structBody '}' ;

enumDeclaration : 'enum' IDENTIFIER '{' enumBody '}' ;

baseList : ':' type (',' type)* ;

classBody : (classMemberDeclaration)* ;

interfaceBody : (interfaceMemberDeclaration)* ;

structBody : (structMemberDeclaration)* ;

enumBody : enumMember (',' enumMember)* ;

classMemberDeclaration : methodDeclaration | fieldDeclaration ;

interfaceMemberDeclaration : methodDeclaration ;

structMemberDeclaration : methodDeclaration | fieldDeclaration ;

methodDeclaration : modifiers? returnType IDENTIFIER '(' parameterList? ')' block ;

modifiers : 'public' | 'private' | 'protected' | 'internal' | 'static' ;

fieldDeclaration : type IDENTIFIER ('=' expression)? ';' ;

enumMember : IDENTIFIER ('=' expression)? ;

returnType : type | 'void' ;

type : qualifiedIdentifier ;

qualifiedIdentifier : IDENTIFIER ('.' IDENTIFIER)* ;

typeParameterList : IDENTIFIER (',' IDENTIFIER)* ;

parameterList : parameter (',' parameter)* ;

parameter : type IDENTIFIER
          | type '[' ']' IDENTIFIER ;

block : '{' statement* '}' ;

statement : localVariableDeclaration
          | expressionStatement
          | returnStatement
          | ifStatement
          | whileStatement
          | forStatement
          | functionCallStatement ;

localVariableDeclaration : type IDENTIFIER ('=' expression)? ';' ;

expressionStatement : expression ';' ;

returnStatement : 'return' expression? ';' ;

ifStatement : 'if' '(' expression ')' block ('else' block)? ;

whileStatement : 'while' '(' expression ')' block ;

forStatement : 'for' '(' (localVariableDeclaration | expressionStatement)? expression? ';' expression? ')' block ;

expression : assignmentExpression 
           | methodCall
           ;

assignmentExpression : IDENTIFIER '=' expression
                     | conditionalExpression ;

conditionalExpression : logicalOrExpression ;

logicalOrExpression : logicalAndExpression ( '||' logicalAndExpression )* ;

logicalAndExpression : equalityExpression ( '&&' equalityExpression )* ;

equalityExpression : relationalExpression (('==' | '!=') relationalExpression)* ;

relationalExpression : additiveExpression (('>' | '<' | '>=' | '<=') additiveExpression)* ;

additiveExpression : multiplicativeExpression (('+'|'-') multiplicativeExpression)* ;

multiplicativeExpression : unaryExpression (('*'|'/') unaryExpression)* ;

unaryExpression : postfixExpression
                | ('+'|'-'|'++'|'--') unaryExpression ;

postfixExpression : primaryExpression ('++' | '--')* ;

primaryExpression : IDENTIFIER
                  | LITERAL
                  | '[' .*? ']'
                  | '(' expression ')'
                  ;

methodCall : memberAccess '(' (expression (',' expression)*)? ')' ;

memberAccess : IDENTIFIER ('.' IDENTIFIER)* ;

functionCallStatement : memberAccess '(' (expression (',' expression)*)? ')' ';' ;

attributeList : '[' (attribute (',' attribute)*)? ']' ;

attribute : IDENTIFIER ('.' IDENTIFIER)* ('(' attributeArgumentList? ')')? ;

attributeArgumentList : attributeArgument (',' attributeArgument)* ;

attributeArgument : IDENTIFIER '=' (IDENTIFIER | LITERAL) ;

comparisonOperator : '>' | '<' | '>=' | '<=' | '==' | '!=' ;

IDENTIFIER : [a-zA-Z_][a-zA-Z_0-9]* ;
LITERAL : [0-9]+ | '"' .*? '"' | '\'' . '\'' ;
WS : [ \t\r\n]+ -> skip ;