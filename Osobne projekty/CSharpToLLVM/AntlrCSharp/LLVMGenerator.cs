using System.Text;

namespace AntlrCSharp;

public class LLVMGenerator : CSharpBaseVisitor<string>
    {
        private StringBuilder llvmIRCode = new StringBuilder();

        public string Generate(CSharpParser.CompilationUnitContext tree)
        {
            llvmIRCode.AppendLine("declare void @Console.WriteLine(i8*)");
            Visit(tree);
            llvmIRCode.AppendLine();
            llvmIRCode.AppendLine("@str.Function = private unnamed_addr constant [9 x i8] c\"Function\\00\"");
            return llvmIRCode.ToString();
        }

        public override string VisitCompilationUnit(CSharpParser.CompilationUnitContext context)
        {
            foreach (var child in context.children)
            {
                Visit(child);
            }
            return null;
        }

        public override string VisitClassDeclaration(CSharpParser.ClassDeclarationContext context)
        {
            llvmIRCode.AppendLine("; Class declaration");
            llvmIRCode.AppendLine($"%{context.IDENTIFIER().GetText()} = type {{ }}");
            return base.VisitClassDeclaration(context);
        }

        public override string VisitMethodDeclaration(CSharpParser.MethodDeclarationContext context)
        {
            llvmIRCode.AppendLine("; Method declaration");
            var returnType = "void"; // Changed to void for simplicity
            var methodName = context.IDENTIFIER().GetText();
            if (methodName == "Main")
            {
                methodName = "main"; // Setting the main entry point
            }
            llvmIRCode.AppendLine($"define dso_local {returnType} @{methodName}() {{");
            llvmIRCode.AppendLine("entry:");
            Visit(context.block());
            llvmIRCode.AppendLine("ret void");
            llvmIRCode.AppendLine("}");
            return null;
        }

        public override string VisitBlock(CSharpParser.BlockContext context)
        {
            foreach (var statement in context.statement())
            {
                Visit(statement);
            }
            return null;
        }

        public override string VisitLocalVariableDeclaration(CSharpParser.LocalVariableDeclarationContext context)
        {
            // Implementacja zmiennych lokalnych (pomijamy w tym przykładzie dla uproszczenia)
            return null;
        }

        public override string VisitExpressionStatement(CSharpParser.ExpressionStatementContext context)
        {
            var expression = Visit(context.expression());
            llvmIRCode.AppendLine(expression);
            return null;
        }

        public override string VisitReturnStatement(CSharpParser.ReturnStatementContext context)
        {
            if (context.expression() != null)
            {
                var returnValue = Visit(context.expression());
                llvmIRCode.AppendLine($"ret {returnValue}");
            }
            else
            {
                llvmIRCode.AppendLine("ret void");
            }
            return null;
        }

        public override string VisitIfStatement(CSharpParser.IfStatementContext context)
        {
            var condition = Visit(context.expression());
            llvmIRCode.AppendLine($"br i1 {condition}, label %if_true, label %if_false");

            // Generowanie kodu dla bloku "if_true"
            llvmIRCode.AppendLine("if_true:");
            Visit(context.block());
            llvmIRCode.AppendLine("br label %if_end");

            // Generowanie kodu dla bloku "if_false"
            llvmIRCode.AppendLine("if_false:");
            if (context.GetChild(5) != null && context.GetChild(5) is CSharpParser.BlockContext elseBlock)
            {
                Visit(elseBlock);
            }
            llvmIRCode.AppendLine("br label %if_end");

            // Label "if_end"
            llvmIRCode.AppendLine("if_end:");
            return null;
        }

        public override string VisitWhileStatement(CSharpParser.WhileStatementContext context)
        {
            llvmIRCode.AppendLine("br label %while_cond");
            llvmIRCode.AppendLine("while_cond:");
            var condition = Visit(context.expression());
            llvmIRCode.AppendLine($"br i1 {condition}, label %while_body, label %while_end");
            llvmIRCode.AppendLine("while_body:");
            Visit(context.block());
            llvmIRCode.AppendLine("br label %while_cond");
            llvmIRCode.AppendLine("while_end:");
            return null;
        }

        public override string VisitForStatement(CSharpParser.ForStatementContext context)
        {
            if (context.localVariableDeclaration() != null)
            {
                Visit(context.localVariableDeclaration());
            }
            else if (context.expression(0) != null)
            {
                Visit(context.expression(0));
            }

            llvmIRCode.AppendLine("br label %for_cond");
            llvmIRCode.AppendLine("for_cond:");
            var condition = Visit(context.expression(1));
            llvmIRCode.AppendLine($"br i1 {condition}, label %for_body, label %for_end");
            llvmIRCode.AppendLine("for_body:");
            Visit(context.block());
            if (context.expression(2) != null)
            {
                Visit(context.expression(2));
            }
            llvmIRCode.AppendLine("br label %for_cond");
            llvmIRCode.AppendLine("for_end:");
            return null;
        }

        public override string VisitExpression(CSharpParser.ExpressionContext context)
        {
            if (context.assignmentExpression() != null)
            {
                return Visit(context.assignmentExpression());
            }
            else if (context.methodCall() != null)
            {
                return Visit(context.methodCall());
            }
            return string.Empty;
        }

        public override string VisitAssignmentExpression(CSharpParser.AssignmentExpressionContext context)
        {
            if (context.IDENTIFIER() != null)
            {
                var identifier = context.IDENTIFIER().GetText();
                var value = Visit(context.expression());
                var type = "i32"; // Przykład dla int
                llvmIRCode.AppendLine($"store {type} {value}, {type}* %{identifier}");
                return value;
            }
            else if (context.conditionalExpression() != null)
            {
                return Visit(context.conditionalExpression());
            }
            return string.Empty;
        }

        public override string VisitConditionalExpression(CSharpParser.ConditionalExpressionContext context)
        {
            return Visit(context.logicalOrExpression());
        }

        public override string VisitLogicalOrExpression(CSharpParser.LogicalOrExpressionContext context)
        {
            var left = Visit(context.logicalAndExpression(0));
            for (int i = 1; i < context.logicalAndExpression().Length; i++)
            {
                var right = Visit(context.logicalAndExpression(i));
                llvmIRCode.AppendLine($"or i1 {left}, {right}");
            }
            return left;
        }

        public override string VisitLogicalAndExpression(CSharpParser.LogicalAndExpressionContext context)
        {
            var left = Visit(context.equalityExpression(0));
            for (int i = 1; i < context.equalityExpression().Length; i++)
            {
                var right = Visit(context.equalityExpression(i));
                llvmIRCode.AppendLine($"and i1 {left}, {right}");
            }
            return left;
        }

        public override string VisitEqualityExpression(CSharpParser.EqualityExpressionContext context)
        {
            var left = Visit(context.relationalExpression(0));
            for (int i = 1; i < context.relationalExpression().Length; i++)
            {
                var right = Visit(context.relationalExpression(i));
                if (context.GetChild(i - 1).GetText() == "==")
                {
                    llvmIRCode.AppendLine($"icmp eq {left}, {right}");
                }
                else if (context.GetChild(i - 1).GetText() == "!=")
                {
                    llvmIRCode.AppendLine($"icmp ne {left}, {right}");
                }
            }
            return left;
        }

        public override string VisitRelationalExpression(CSharpParser.RelationalExpressionContext context)
        {
            var left = Visit(context.additiveExpression(0));
            for (int i = 1; i < context.additiveExpression().Length; i++)
            {
                var right = Visit(context.additiveExpression(i));
                var op = context.GetChild(i - 1).GetText();
                if (op == ">")
                {
                    llvmIRCode.AppendLine($"icmp sgt {left}, {right}");
                }
                else if (op == "<")
                {
                    llvmIRCode.AppendLine($"icmp slt {left}, {right}");
                }
                else if (op == ">=")
                {
                    llvmIRCode.AppendLine($"icmp sge {left}, {right}");
                }
                else if (op == "<=")
                {
                    llvmIRCode.AppendLine($"icmp sle {left}, {right}");
                }
            }
            return left;
        }

        public override string VisitAdditiveExpression(CSharpParser.AdditiveExpressionContext context)
        {
            var left = Visit(context.multiplicativeExpression(0));
            for (int i = 1; i < context.multiplicativeExpression().Length; i++)
            {
                var right = Visit(context.multiplicativeExpression(i));
                var op = context.GetChild(i - 1).GetText();
                if (op == "+")
                {
                    llvmIRCode.AppendLine($"add {left}, {right}");
                }
                else if (op == "-")
                {
                    llvmIRCode.AppendLine($"sub {left}, {right}");
                }
            }
            return left;
        }

        public override string VisitMultiplicativeExpression(CSharpParser.MultiplicativeExpressionContext context)
        {
            var left = Visit(context.unaryExpression(0));
            for (int i = 1; i < context.unaryExpression().Length; i++)
            {
                var right = Visit(context.unaryExpression(i));
                var op = context.GetChild(i - 1).GetText();
                if (op == "*")
                {
                    llvmIRCode.AppendLine($"mul {left}, {right}");
                }
                else if (op == "/")
                {
                    llvmIRCode.AppendLine($"div {left}, {right}");
                }
            }
            return left;
        }

        public override string VisitUnaryExpression(CSharpParser.UnaryExpressionContext context)
        {
            if (context.postfixExpression() != null)
            {
                return Visit(context.postfixExpression());
            }
            else if (context.GetChild(0).GetText() == "+" || context.GetChild(0).GetText() == "-" || context.GetChild(0).GetText() == "++" || context.GetChild(0).GetText() == "--")
            {
                var unaryExpression = Visit(context.unaryExpression());
                var op = context.GetChild(0).GetText();
                if (op == "+")
                {
                    return unaryExpression; // plus nie zmienia wartości
                }
                else if (op == "-")
                {
                    llvmIRCode.AppendLine($"neg {unaryExpression}");
                    return unaryExpression;
                }
                else if (op == "++")
                {
                    llvmIRCode.AppendLine($"add {unaryExpression}, 1");
                    return unaryExpression;
                }
                else if (op == "--")
                {
                    llvmIRCode.AppendLine($"sub {unaryExpression}, 1");
                    return unaryExpression;
                }
            }
            return string.Empty;
        }

        public override string VisitPostfixExpression(CSharpParser.PostfixExpressionContext context)
        {
            if (context.primaryExpression() != null)
            {
                return Visit(context.primaryExpression());
            }
            return string.Empty;
        }
        
        public override string VisitPrimaryExpression(CSharpParser.PrimaryExpressionContext context)
        {
            if (context.IDENTIFIER() != null)
            {
                return $"%{context.IDENTIFIER().GetText()}";
            }
            else if (context.LITERAL() != null)
            {
                return context.LITERAL().GetText();
            }
            else if (context.expression() != null)
            {
                return Visit(context.expression());
            }
            return string.Empty;
        }

        public override string VisitMethodCall(CSharpParser.MethodCallContext context)
        {
            var methodName = context.memberAccess().GetText();
            var arguments = string.Join(", ", context.expression().Select(e => Visit(e)));
            if (methodName == "Console.WriteLine")
            {
                arguments = "i8* getelementptr inbounds ([9 x i8], [9 x i8]* @str.Function, i32 0, i32 0)";
            }
            return $"call void @{methodName}({arguments})";
        }
    }