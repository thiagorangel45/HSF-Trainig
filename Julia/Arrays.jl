@show my_list = [1, 2, 'a'] #you can make "Python like" arrays with mixed types in them

@show my_integer_array = [1,2,7, 66] #but Julia will specialise an Array literal with only 1 type in it to be uniform

@show typeof(my_integer_array)

@show my_float_array = Float64[1, 5.5, 5//6] #or we can explicitly impose a type

@show squared_integers = [i^2 for i in 1:10];

@show my_list[1] == my_list[begin] #basic indexing

@show my_integer_array[3:end] #last 2 elements

mask = [ iseven(i) for i in 1:10 ] #only even values are true
@show squared_integers[mask]; #selects only elements in the mask

@show reflection = [ [1,0] [0,-1] ]

"""The top-left element of reflection is $(reflection[1,1]) or $(reflection[begin,begin])"""

threed_array = [ 1 ; 0 ;; 2 ; 2 ;;; 3 ; 3 ;; 0 ; 1 ] 

odd_powers_and_offset = [ i^j + k for i in 0:3, j in 1:2:5, k in 0:1 ] 

@show many_zeros = zeros(Float64, 2, 4, 3) #2x4x3 array, zero'd

m = [0,1,2,3]

reshape(m, (2,2)) #2x2 matrix, column-major!

slice1 = odd_powers_and_offset[:,:,1]
slice2 = odd_powers_and_offset[:,:,1]

@show slice1 === slice2 #false, because these are both different copies of the underlying data

slice1[end,end,end] = 88

slice2[end,end,end] #hasn't changed because these are different memory

view1 = view(odd_powers_and_offset, :, : , 1)
view2 = @view odd_powers_and_offset[:,:,begin]
@views view3 = odd_powers_and_offset[:,:,begin]   

@show view1
@show view1 === view2 === view3 #true, because these reference the same memory

view1[end,end,end] = 88

odd_powers_and_offset[end,end,1] #has changed, as these are the same memory


