mycursor.execute("create table BC_DATA (id int auto_increment primary key) ")
mycursor.execute("drop table BC_DATA")
mycursor.execute("create table BC_DATA (id int(255))")
for header in headers:
    if header != "id":
        mycursor.execute(f"alter table BC_DATA add {header} FLOAT ")
