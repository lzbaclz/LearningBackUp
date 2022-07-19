# MySQL Learning Tutorial

## Basement
- 创建数据表: CREATE TABLE table_name (column_name column_type);

(删)删除数据表: DROP TABLE table_name ;

(增)插入数据: INSERT INTO table_name ( field1, field2,...fieldN )
            VALUES
            ( value1, value2,...valueN );

(查)查询数据: SELECT column_name,column_name
            FROM table_name
            [WHERE Clause]
            [LIMIT N][ OFFSET M]

(改)更新数据: UPDATE table_name SET field1=new-value1, field2=new-value2
            [WHERE Clause]
Note 将字段中的特定字符串批量修改为其他字符串
UPDATE table_name SET field=REPLACE(field, 'old-string', 'new-string') 
[WHERE Clause]

(删)删除数据: DELETE FROM table_name [WHERE Clause]
Note: 
1. delete 和 truncate 仅仅删除表数据，drop 连表数据和表结构一起删除
2. delete 是 DML 语句，操作完以后如果没有不想提交事务还可以回滚，truncate 和 drop 是 DDL 语句，操作完马上生效，不能回滚




- WHERE 子句
SELECT field1, field2,...fieldN FROM table_name1, table_name2...
[WHERE condition1 [AND [OR]] condition2.....

where：数据库中常用的是where关键字，用于在初始表中筛选查询。它是一个约束声明，用于约束数据，在返回结果集之前起作用。

group by:对select查询出来的结果集按照某个字段或者表达式进行分组，获得一组组的集合，然后从每组中取出一个指定字段或者表达式的值。

having：用于对where和group by查询出来的分组经行过滤，查出满足条件的分组结果。它是一个过滤声明，是在查询返回结果集以后对查询结果进行的过滤操作。

执行顺序 select –> where –> group by –> having –> order by


- LIKE 子句
使用百分号 %字符来表示任意字符，类似于UNIX或正则表达式中的星号 *。没有使用百分号 %, LIKE 子句与等号 = 的效果是一样的

SELECT field1, field2,...fieldN 
FROM table_name
WHERE field1 LIKE condition1 [AND [OR]] filed2 = 'somevalue'

模糊匹配 % 和 _ 
%：表示任意 0 个或多个字符。可匹配任意类型和长度的字符，有些情况下若是中文，请使用两个百分号（%%）表示。
_：表示任意单个字符。匹配单个任意字符，它常用来限制表达式的字符长度语句。


- UNION 操作符
用于连接两个以上的 SELECT 语句的结果组合到一个结果集合中。多个 SELECT 语句会删除重复的数据。

SELECT expression1, expression2, ... expression_n
FROM tables
[WHERE conditions]
UNION [ALL | DISTINCT]
SELECT expression1, expression2, ... expression_n
FROM tables
[WHERE conditions];

DISTINCT: 可选，删除结果集中重复的数据。默认情况下 UNION 操作符已经删除了重复数据，所以 DISTINCT 修饰符对结果没啥影响。
ALL: 可选，返回所有结果集，包含重复数据。


- MySQL 排序
ORDER BY 子句来设定你想按哪个字段哪种方式来进行排序，再返回搜索结果。

SELECT field1, field2,...fieldN FROM table_name1, table_name2...
ORDER BY field1 [ASC [DESC][默认 ASC]], [field2...] [ASC [DESC][默认 ASC]]


- 分组
GROUP BY 语句，在分组的列上我们可以使用 COUNT, SUM, AVG,等函数。

SELECT column_name, function(column_name)
FROM table_name
WHERE column_name operator value
GROUP BY column_name;




- **MySQL 连接**
SELECT, UPDATE 和 DELETE 语句中使用 Mysql 的 JOIN 来联合多表查询

分类
    - INNER JOIN（内连接,或等值连接）：获取两个表中字段匹配关系的记录。
    - LEFT JOIN（左连接）：获取左表所有记录，即使右表没有对应匹配的记录。
    - RIGHT JOIN（右连接）： 与 LEFT JOIN 相反，用于获取右表所有记录，即使左表没有对应匹配的记录。

1. SELECT a.runoob_id, a.runoob_author, b.runoob_count FROM runoob_tbl a INNER JOIN tcount_tbl b ON a.runoob_author = b.runoob_author;
2. SELECT a.runoob_id, a.runoob_author, b.runoob_count FROM runoob_tbl a LEFT JOIN tcount_tbl b ON a.runoob_author = b.runoob_author;
3. SELECT a.runoob_id, a.runoob_author, b.runoob_count FROM runoob_tbl a RIGHT JOIN tcount_tbl b ON a.runoob_author = b.runoob_author;


- NULL 值处理
MySQL 使用 SQL SELECT 命令及 WHERE 子句来读取数据表中的数据,但是当提供的查询条件字段为 NULL 时，该命令可能就无法正常工作。
    - IS NULL: 当列的值是 NULL,此运算符返回 true。
    - IS NOT NULL: 当列的值不为 NULL, 运算符返回 true。
    - <=>: 比较操作符（不同于 = 运算符），当比较的的两个值相等或者都为 NULL 时返回 true。





--- 
Exercises:
REGEXP: [[:digit:]]{4}


