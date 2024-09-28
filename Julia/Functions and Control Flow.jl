function mymult(a, b)
    a * b
end

mymult(3, 7)

mymult(6+3im, -9-7im)

mymult("gold", "leader")

"""
    adder(a, b)
    This function adds `a` and `b`
"""
function adder(a, b)
    a+b
end

?adder

mypow(a, b) = a^b

mypow(2, 8)

mypow("old ", 4)

x -> x>2

map(x -> x>2, 1:5)

function gtlteq(a, b)
    if a > b
        println("$a is greater than $b")
    elseif a < b
        println("$a is less than $b")
    elseif a == b
        println("$a is equal to $b")
    else
        println("I have no idea about $a and $b!")
    end
end

gtlteq(2.3, -9)

gtlteq("apples", "oranges")

true && "it's true"

false || "it's false"

false && "it's true"

2 > 4 ? "it's bigger" : "it's smaller"

for number in 1:4
    println("I am at number $number")
end

for (index, string) in enumerate(["I", "think", "therefore", "I", "am"])
    println("Word $index is $string")
end

for (hip, hop, hup) ∈ zip(1:3, 10:12, ["yes", "no", "maybe"])
    println("The $hip and $hop say $hup")
end

for i in 1:3, j in 1:4  #like for i=1:3 ; for j=1:4 ...
    println("The product of $i and $j is $(i*j)")
end

countdown = 10
while countdown > 0
    println(countdown, "...")
    countdown -= 1
end
println("blast off!")


