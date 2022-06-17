# SQL样式指南 · SQL Style Guide:
<https://www.sqlstyle.guide/zh/>

# Basement
## 01 存储
-- 创建数据库
CREATE DATABASE demo；
-- 删除数据库
DROP DATABASE demo；
-- 查看数据库
SHOW DATABASES;
-- 创建数据表：
CREATE TABLE demo.test
(  
  barcode text,
  goodsname text,
  price int
); 
-- 查看表结构
DESCRIBE demo.test;
-- 查看所有表
SHOW TABLES;
-- 添加主键
ALTER TABLE demo.test
ADD COLUMN itemnumber int PRIMARY KEY AUTO_INCREMENT;
-- 向表中添加数据
INSERT INTO demo.test
(barcode,goodsname,price)
VALUES ('0001','本',3);

## 02 字段
-- 修改字段类型语句
ALTER TABLE demo.goodsmaster
MODIFY COLUMN price DOUBLE;
-- 计算字段合计函数：
SELECT SUM(price)
FROM demo.goodsmaster;

Data Types <https://dev.mysql.com/doc/refman/8.0/en/data-types.html>


## 03 表

CREATE TABLE <表名>
(
字段名1 数据类型 [字段级别约束] [默认值]，
字段名2 数据类型 [字段级别约束] [默认值]，
......
[表级别约束]
);


唯一性约束: 表示这个字段的值不能重复，否则系统会提示错误。跟主键约束相比，唯一性约束要更加弱一些。
在一个表中，我们可以指定多个字段满足唯一性约束，而主键约束则只能有一个，这也是 MySQL 系统决定的。



- 复制表
CREATE TABLE demo.importheadhist
LIKE demo.importhead;


- 给表增加列
mysql> ALTER TABLE demo.importheadhist
    -> ADD confirmer INT; -- 添加一个字段confirmer，类型INT


mysql> ALTER TABLE demo.importheadhist
    -> ADD confirmdate DATETIME; -- 添加一个字段confirmdate，类型是DATETIME


- 修改列(名称，格式)
mysql> ALTER TABLE demo.importheadhist
    -> CHANGE quantity importquantity DOUBLE;


- 不想改变字段名称，只想改变字段类型
ALTER TABLE demo.importheadhist
MODIFY importquantity DECIMAL(10,3);


- 向表中添加一个字段
ALTER TABLE demo.importheadhist
ADD suppliername TEXT AFTER supplierid;


总结
约束，包括默认约束、非空约束、唯一性约束和自增约束等。

默认值约束：就是给字段设置一个默认值。
非空约束：就是声明字段不能为空值。
唯一性约束：就是声明字段不能重复。
自增约束：就是声明字段值能够自动加 1，且不会重复。


CREATE TABLE
(
字段名 字段类型 PRIMARY KEY
);
CREATE TABLE
(
字段名 字段类型 NOT NULL
);
CREATE TABLE
(
字段名 字段类型 UNIQUE
);
CREATE TABLE
(
字段名 字段类型 DEFAULT 值
);
-- 这里要注意自增类型的条件，字段类型必须是整数类型。
CREATE TABLE
(
字段名 字段类型 AUTO_INCREMENT
);
-- 在一个已经存在的表基础上，创建一个新表
CREATE TABLE demo.importheadhist LIKE demo.importhead;
-- 修改表的相关语句
ALTER TABLE 表名 CHANGE 旧字段名 新字段名 数据类型;
ALTER TABLE 表名 ADD COLUMN 字段名 字段类型 FIRST|AFTER 字段名;
ALTER TABLE 表名 MODIFY 字段名 字段类型 FIRST|AFTER 字段名;



MySQL 的官方文档
CREATE TABLE Statement: <https://dev.mysql.com/doc/refman/8.0/en/create-table.html>
ALTER TABLE Statement: <https://dev.mysql.com/doc/refman/8.0/en/alter-table.html>




## 04增删查改
部分插入一条数据记录是可以的，但前提是，没有赋值的字段，一定要让 MySQL 知道如何处理，比如可以为空、有默认值，或者是自增约束字段，等等

- 插入查询结果
INSERT INTO 表名 （字段名）
SELECT 字段名或值
FROM 表名
WHERE 条件

- 删除数据
DELETE FROM 表名
WHERE 条件

删除全部数据 DELETE FROM [tablename]; 会报错
Workbench 自动处于安全模式，它要求对数据的删除或修改操作中必须包含 WHERE 条件。而且，这个 WHERE 条件中，必须用到主键约束或者唯一性约束的字段。


- 修改数据
UPDATE 表名
SET 字段名=值
WHERE 条件

Note: 不要修改主键字段的值


- 查询数据
SELECT *|字段列表
FROM 数据源
WHERE 条件
GROUP BY 字段
HAVING 条件
ORDER BY 字段
LIMIT 起始点，行数



INSERT INTO 表名 [(字段名 [,字段名] ...)] VALUES (值的列表);
 
INSERT INTO 表名 （字段名）
SELECT 字段名或值
FROM 表名
WHERE 条件
 
DELETE FROM 表名
WHERE 条件
 
UPDATE 表名
SET 字段名=值
WHERE 条件

SELECT *|字段列表
FROM 数据源
WHERE 条件
GROUP BY 字段
HAVING 条件
ORDER BY 字段
LIMIT 起始点，行数


INSERT Statement: <https://dev.mysql.com/doc/refman/8.0/en/insert.html>
UPDATE Statement: <https://dev.mysql.com/doc/refman/8.0/en/update.html>
SELECT Statement: <https://dev.mysql.com/doc/refman/8.0/en/select.html>




# 05 主键
三种设置主键的思路：
    业务字段做主键（不建议）
    自增字段做主键 (只适用于单机系统--如果有两个子表的PRIMARY KEY的字段是相同的，上传到主表的时候KEY可能会有冲突)
    手动赋值字段做主键（最好，同时也会复杂）


案例：
比如，A 店的 MySQL 数据库中的 demo.membermaster 中，字段“id”的值是 100，这个时候，新增了一个会员，“id”是 101。同时，B 店的字段“id”值也是 100，要加一个新会员，“id”也是 101，毕竟，B 店的 MySQL 数据库与 A 店相互独立。等 A 店与 B 店都把新的会员上传到总部之后，就会出现两个“id”是 101，但却是不同会员的情况，这该如何处理呢？

具体的操作是这样的：在总部 MySQL 数据库中，有一个管理信息表，里面的信息包括成本核算策略，支付方式等，还有总部的系统参数，我们可以在这个表中添加一个字段，专门用来记录当前会员编号的最大值。


Note: 更改主键设置的成本非常高






## 06 外键和连接：如何做关联查询
这种把分散在多个不同的表里的数据查询出来的操作，就是多表查询

为了把 2 个表关联起来，会用到 2 个重要的功能：外键（FOREIGN KEY）和连接（JOIN）。

外键约束定义的语法结构：
[CONSTRAINT <外键约束名称>] FOREIGN KEY 字段名
REFERENCES <主表名> 字段名


CREATE TABLE 从表名
(
  字段名 类型,
  ...
-- 定义外键约束，指出外键字段和参照的主表字段
CONSTRAINT 外键约束名
FOREIGN KEY (字段名) REFERENCES 主表名 (字段名)
)


可以通过修改表来定义外键约束：
ALTER TABLE 从表名 
ADD CONSTRAINT 约束名 FOREIGN KEY 字段名 
REFERENCES 主表名 （字段名）;


CREATE TABLE demo.importdetails
( 
    listnumber INT, 
    itemnumber INT, 
    quantity DECIMAL(10,3), 
    importprice DECIMAL(10,2), 
    importvalue DECIMAL(10,2),  
    CONSTRAINT fk_importdetails_importhead 
    FOREIGN KEY listnumber <-- 没有在FOREIGN KEY后面添加'()'
    REFERENCES importhead (listnumber)
);



CREATE TABLE demo.importdetails
    -> (
    ->   listnumber INT,
    ->   itemnumber INT,
    ->   quantity DECIMAL(10,3),
    ->   importprice DECIMAL(10,2),
    ->   importvalue DECIMAL(10,2),
    ->   -- 定义外键约束，指出外键字段和参照的主表字段
    ->   CONSTRAINT fk_importdetails_importhead
    ->   FOREIGN KEY (listnumber) REFERENCES importhead (listnumber)
    -> );

--- 约束的查询
SELECT 
constraint_name, -- 表示外键约束名称 
table_name, -- 表示外键约束所属数据表的名称 
column_name, -- 表示外键约束的字段名称 
referenced_table_name, -- 表示外键约束所参照的数据表名称 
referenced_column_name -- 表示外键约束所参照的字段名称 
FROM 
information_schema.KEY_COLUMN_USAGE 
WHERE 
constraint_name = 'fk_importdetails_importhead';





-- 定义外键约束：
CREATE TABLE 从表名
(
字段 字段类型
....
CONSTRAINT 外键约束名称
FOREIGN KEY (字段名) REFERENCES 主表名 (字段名称)
);
ALTER TABLE 从表名 ADD CONSTRAINT 约束名 FOREIGN KEY 字段名 REFERENCES 主表名 （字段名）;

-- 连接查询
SELECT 字段名
FROM 表名 AS a
JOIN 表名 AS b
ON (a.字段名称=b.字段名称);
 
SELECT 字段名
FROM 表名 AS a
LEFT JOIN 表名 AS b
ON (a.字段名称=b.字段名称);
 
SELECT 字段名
FROM 表名 AS a
RIGHT JOIN 表名 AS b
ON (a.字段名称=b.字段名称);


总结：外键(FOREIGN KEY)的存在是为了在两个表之间加一个隐形的联系。保证了1）数据的添加与删除是有顺序的。 2）数据的错误顺序删除或者添加会报错。
虽然在实际的链接查询中可能不一定需要外键（高并发的sql查询设置外键会大大降低查询的效率），但是外键的存在让表格之间的关联变得更加安全

添加了外键的表格的数据的添加与删除类似--堆--先入后出
先添加主键的表格的内容，再添加外键的表格的内容
先删除外键的表格的内容，再删除主键的表格的内容


--- 思考题：根据table_b的一条信息改变table_a的数据，模板
update table_a as a, table_b as b
set a.columnname = new_value
where a.joinelment = b.joinelment
and b.comlumnname [requirements]







## 07 条件语句，where & having

HAVING 则需要跟分组关键字 GROUP BY 一起使用，通过对分组字段或分组计算函数进行限定，来筛选结果


--- exe
select * 
from demo.transactionhead as A
join demo.transactiondetails as B on (A.transactionid = B.transactionid)
join demo.operator as C on (A.operatorid = C.operatorid);



## 08 聚合函数

求和函数 SUM()、求平均函数 AVG()、最大值函数 MAX()、最小值函数 MIN() 和计数函数 COUNT()

LEFT(str，n)：表示返回字符串 str 最左边的 n 个字符

COUNT（字段）用来统计分组内这个字段的值出现了多少次。如果字段值是空，就不统计。



## 09 时间函数

- 获取日期时间数据中部分信息的函数
要统计一天中每小时的销售情况，这就要用到 MySQL 的日期时间处理函数 EXTRACT（）和 HOUR（）了。
EXTRACT（type FROM date）--> EXTRACT(HOUR FROM b.transdate) == HOUR(b.transdate)

用“HOUR”提取时间类型 DATETIME 中的小时信息，同样道理，可以用“YEAR”获取年度信息，用“MONTH”获取月份信息，用“DAY”获取日的信息。
时间单位：<https://dev.mysql.com/doc/refman/8.0/en/expressions.html#temporal-intervals>


YEAR（date）：获取 date 中的年。
MONTH（date）：获取 date 中的月。
DAY（date）：获取 date 中的日。
HOUR（date）：获取 date 中的小时。
MINUTE（date）：获取 date 中的分。
SECOND（date）：获取 date 中的秒。


- 计算日期时间的函数

2 个常用的 MySQL 的日期时间计算函数。

DATE_ADD（date, INTERVAL 表达式 type）：表示计算从时间点“date”开始，向前或者向后一段时间间隔的时间。“表达式”的值为时间间隔数，正数表示向后，负数表示向前，“type”表示时间间隔的单位（比如年、月、日等）。

LAST_DAY（date）：表示获取日期时间“date”所在月份的最后一天的日期。

用 DATE_ADD 函数，获取到 2020 年 12 月 10 日上一年的日期：2019 年 12 月 10 日。

DATE_ADD('2020-12-10', INTERVAL - 1 YEAR);


mysql>  SELECT DATE_ADD(DATE_ADD('2020-12-10', INTERVAL - 1 YEAR),INTERVAL - 1 MONTH);
+------------------------------------------------------------------------+
| DATE_ADD(DATE_ADD('2020-12-10', INTERVAL - 1 YEAR),INTERVAL - 1 MONTH) |
+------------------------------------------------------------------------+
| 2019-11-10                                                             |
+------------------------------------------------------------------------+
1 row in set (0.00 sec)

除了 DATE_ADD()，ADDDATE()、DATE_SUB() 和 SUBDATE() 也能达到同样的效果。
ADDDATE()：跟 DATE_ADD() 用法一致；
DATE_SUB()，SUBDATE()：与 DATE_ADD() 用法类似，方向相反，执行日期的减操作。


- 其他日期时间函数

MySQL 中 CASE 函数的语法如下：
CASE 表达式 WHEN 值1 THEN 表达式1 [ WHEN 值2 THEN 表达式2] ELSE 表达式m END

DATE_FORMAT()，它表示将日期时间“date”按照指定格式显示。
DATE_FORMAT(date,format): <https://dev.mysql.com/doc/refman/8.0/en/date-and-time-functions.html#function_date-format>

用 24 小时制：
mysql> SELECT DATE_FORMAT("2020-12-01 13:25:50","%T");
+-----------------------------------------+
| DATE_FORMAT("2020-12-01 13:25:50","%T") |
+-----------------------------------------+
| 13:25:50                                |
+-----------------------------------------+
1 row in set (0.00 sec)


按照上下午的方式来查看时间：
mysql> SELECT DATE_FORMAT("2020-12-01 13:25:50","%r");
+-----------------------------------------+
| DATE_FORMAT("2020-12-01 13:25:50","%r") |
+-----------------------------------------+
| 01:25:50 PM                             |
+-----------------------------------------+
1 row in set (0.00 sec


DATEDIFF（date1,date2），表示日期“date1”与日期“date2”之间差几天

mysql> SELECT DATEDIFF("2021-02-01","2020-12-01");
+-------------------------------------+
| DATEDIFF("2021-02-01","2020-12-01") |
+-------------------------------------+
|                                  62 |
+-------------------------------------+
1 row in set (0.00 sec)


--- 思考题假如用户想查一下今天是星期几（不能用数值，要用英文显示），你可以写一个简单的查询语句吗？
mysql> SELECT DATE_FORMAT(CURRENT_DATE(), '%W');
+-----------------------------------+
| DATE_FORMAT(CURRENT_DATE(), '%W') |
+-----------------------------------+
| Thursday                          |
+-----------------------------------+
1 row in set (0.00 sec)



## 11 数学计算、字符串处理和条件判断

- 数学函数
用来处理数值数据
    取整函数 ROUND()、CEIL()、FLOOR()
    绝对值函数 ABS() 
    求余函数 MOD()


- 字符串函数
    CONCAT（s1,s2,...）：表示把字符串 s1、s2……拼接起来，组成一个字符串。
    CAST（表达式 AS CHAR）：表示将表达式的值转换成字符串。
    CHAR_LENGTH（字符串）：表示获取字符串的长度。
    SPACE（n）：表示获取一个由 n 个空格组成的字符串。




























