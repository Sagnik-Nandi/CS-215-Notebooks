# Function to update mean
def updateMean(oldMean, newData, n, A):
    return (oldMean*n +newData)/(n+1)

# Function to update median
def updateMedian(oldMedian , newData, n ,A):
    if(n==0):return newData
    if(n==1):return float((oldMedian+newData)/2)

    if(n%2==0): 
        if(A[int((n-1)/2)]<=newData and newData<=A[int(n/2)]): return newData 
        elif(newData>=A[int(n/2)]): return A[int(n/2)]
        else: return A[int((n-1)/2)]

    else:
        if(A[int((n/2)-1)]<=newData and newData<=A[int((n/2)+1)]): return float((oldMedian+newData)/2)
        elif(newData>=A[int((n/2)+1)]): return float((oldMedian+A[int((n/2)+1)])/2)
        else: return float((oldMedian+A[int((n/2)-1)])/2)

        
# Function to update std dev
def updateStd(oldStd, oldMean, newData, n, A):

    # Using the identity 
    # var=(sum of squares)/n - (mean)^2
    
    oldVar = oldStd**2
    oldMeanSquare = oldVar + oldMean**2
    newMeanSquare = (oldMeanSquare*n + newData**2)/(n+1)
    newMean = (oldMean*n + newData)/(n+1)
    newVar = newMeanSquare - newMean**2
    return newVar**(1/2)

def test():
    testData = []
    mean=0
    median=0
    std=0
    len=0
    while True:
        print("Enter data: ")
        try :
            testEntry = float(input())
            testData.append(testEntry)

            # Calculations
            std = updateStd(std, mean, testEntry, len, testData)
            mean = updateMean(mean, testEntry, len, testData)
            median = updateMedian(median, testEntry, len,  testData)
            len+=1

            # Show results
            print("Data are ", testData)
            print("Mean is ", mean)
            print("Median is ", median)
            print("Std is ", std, "\n")
        except:
            print("Not a number")
            break


if __name__ == "__main__":
    test()