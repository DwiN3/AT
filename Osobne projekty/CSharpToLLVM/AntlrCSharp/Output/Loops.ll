declare void @Console.WriteLine(i8*)
; Class declaration
%Loops = type { }
; Method declaration
define dso_local void @main() {
entry:
br label %for_cond
for_cond:
br i1 %i, label %for_body, label %for_end
for_body:
call void @Console.WriteLine(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @str.Function, i32 0, i32 0))
br label %for_cond
for_end:
br label %while_cond
while_cond:
br i1 %counter, label %while_body, label %while_end
while_body:
%counter
br label %while_cond
while_end:
ret void
}

@str.Function = private unnamed_addr constant [9 x i8] c"Function\00"
