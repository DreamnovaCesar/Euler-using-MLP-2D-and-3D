def bubbleSort(arr):
    n = len(arr)
    # optimize code, so if the array is already sorted, it doesn't need
    # to go through the entire process
    swapped = False
    # Traverse through all array elements
    for i in range(n-1):
        # range(n) also work but outer loop will
        # repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
         
        if not swapped:
            # if we haven't needed to make a single swap, we
            # can just exit the main loop.
            return
 
 
# Driver code to test above
arr = [     2, 25, 35, 39, 43, 47, 59, 139, 151, 159, 171, 183, 191, 9, 26, 36, 40, 44, 56, 129, 148, 156, 161, 180, 
            188, 11, 27, 37, 41, 45, 57, 131, 149, 157, 163, 181, 189, 24, 33, 38, 42, 46, 58, 137, 150, 158, 169, 182, 190     ]
 
bubbleSort(arr)
 
print("Sorted array is:")
for i in range(len(arr)):
    print("% d" % arr[i], end=",")