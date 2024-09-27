x = 1.09
my_amazing_string = "a string"
δ = 1.0e-9

δ2 = 1.0e-18;

(2^10)

(33%10)

(2^10) + (33%10)

a = 7
b = 13
@show a & b
@show a | b;

a *= 2

a >= 8

a < δ2

a = [1, 2]
b = [1, 2]
c = a
@show a === b
@show a === c

1 < 2 <= 2 < 3 == 3 > 2 >= 1 == 1 < 3 != 5

f = 1.234
typeof(f)

i = UInt(12335124)

typeof(i)

f * i

Int16(i)

integer_value::Int64 = 6;

integer_value = 2.0;

@show integer_value

typeof(integer_value)

integer_value = 2.5

# "im" is the imaginary number constant
m = 1 + 2im
n = -3 - 3im
m*n

# // defines a rational
r1 = 2//3
r2 = 1//4
r1*r2

"here is a string"

"hello " * "world"

текст = """The Ukrainian for "text" is "текст" and the Hindi is "पाठ"!"""

текст[38] == 'т'  #trying to take from position 37 would cause an error, as Cyrillic chars are two-bytes wide.

"$m squared is $(m^2)"

farmyard_sounds = Dict("🐮" => "moo", "sheep" => "baa", "pig" => "oink", "farmer" => "get off my land!")

@show farmyard_sounds["🐮"]

@show haskey(farmyard_sounds,"pig")

@show farmyard_sounds["🐱"] = "Meow" ;

farmyard_sounds

farmyard_sounds["🐮"] = "moo"
haskey(farmyard_sounds, "pig") = true
farmyard_sounds["🐱"] = "Meow" = "Meow"

Dict{String, String} with 5 entries:
  "farmer" => "get off my land!"
  "🐱"     => "Meow"
  "pig"    => "oink"
  "🐮"     => "moo"
  "sheep"  => "baa"

struct my_vector
    x::Float64
    y::Float64
end

υϵκτωρ = my_vector(3.7, 6e7)

struct my_vector
    x::Float64
    y::Float64
end

υϵκτωρ = my_vector(3.7, 6e7)

mutable struct my_mut_vector
    x::Float64
    y::Float64
end

μ_υϵκτωρ = my_mut_vector(3.7, 6e7)

μ_υϵκτωρ.x = 6.0

μ_υϵκτωρ


