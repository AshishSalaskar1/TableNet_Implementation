+
ý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02unknown8¢ó!

block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel

'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0

block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel

'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0

block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock2_conv1/kernel

'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:*
dtype0

block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock2_conv2/kernel

'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv1/kernel

'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:*
dtype0

block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv2/kernel

'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv3/kernel

'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:*
dtype0

block3_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv4/kernel

'block3_conv4/kernel/Read/ReadVariableOpReadVariableOpblock3_conv4/kernel*(
_output_shapes
:*
dtype0
{
block3_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv4/bias
t
%block3_conv4/bias/Read/ReadVariableOpReadVariableOpblock3_conv4/bias*
_output_shapes	
:*
dtype0

block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv1/kernel

'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:*
dtype0

block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv2/kernel

'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:*
dtype0

block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv3/kernel

'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:*
dtype0

block4_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv4/kernel

'block4_conv4/kernel/Read/ReadVariableOpReadVariableOpblock4_conv4/kernel*(
_output_shapes
:*
dtype0
{
block4_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv4/bias
t
%block4_conv4/bias/Read/ReadVariableOpReadVariableOpblock4_conv4/bias*
_output_shapes	
:*
dtype0

block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv1/kernel

'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:*
dtype0

block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv2/kernel

'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:*
dtype0

block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv3/kernel

'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:*
dtype0

block5_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv4/kernel

'block5_conv4/kernel/Read/ReadVariableOpReadVariableOpblock5_conv4/kernel*(
_output_shapes
:*
dtype0
{
block5_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv4/bias
t
%block5_conv4/bias/Read/ReadVariableOpReadVariableOpblock5_conv4/bias*
_output_shapes	
:*
dtype0

block_6_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameblock_6_conv_1/kernel

)block_6_conv_1/kernel/Read/ReadVariableOpReadVariableOpblock_6_conv_1/kernel*(
_output_shapes
:*
dtype0

block_6_conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock_6_conv_1/bias
x
'block_6_conv_1/bias/Read/ReadVariableOpReadVariableOpblock_6_conv_1/bias*
_output_shapes	
:*
dtype0

block_6_conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameblock_6_conv_2/kernel

)block_6_conv_2/kernel/Read/ReadVariableOpReadVariableOpblock_6_conv_2/kernel*(
_output_shapes
:*
dtype0

block_6_conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock_6_conv_2/bias
x
'block_6_conv_2/bias/Read/ReadVariableOpReadVariableOpblock_6_conv_2/bias*
_output_shapes	
:*
dtype0

col_decoder/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecol_decoder/conv2d/kernel

-col_decoder/conv2d/kernel/Read/ReadVariableOpReadVariableOpcol_decoder/conv2d/kernel*(
_output_shapes
:*
dtype0

col_decoder/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namecol_decoder/conv2d/bias

+col_decoder/conv2d/bias/Read/ReadVariableOpReadVariableOpcol_decoder/conv2d/bias*
_output_shapes	
:*
dtype0
«
#col_decoder/conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#col_decoder/conv2d_transpose/kernel
¤
7col_decoder/conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOp#col_decoder/conv2d_transpose/kernel*'
_output_shapes
:*
dtype0

!col_decoder/conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!col_decoder/conv2d_transpose/bias

5col_decoder/conv2d_transpose/bias/Read/ReadVariableOpReadVariableOp!col_decoder/conv2d_transpose/bias*
_output_shapes
:*
dtype0
 
table_decoder/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nametable_decoder/conv2d_1/kernel

1table_decoder/conv2d_1/kernel/Read/ReadVariableOpReadVariableOptable_decoder/conv2d_1/kernel*(
_output_shapes
:*
dtype0

table_decoder/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nametable_decoder/conv2d_1/bias

/table_decoder/conv2d_1/bias/Read/ReadVariableOpReadVariableOptable_decoder/conv2d_1/bias*
_output_shapes	
:*
dtype0
 
table_decoder/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nametable_decoder/conv2d_2/kernel

1table_decoder/conv2d_2/kernel/Read/ReadVariableOpReadVariableOptable_decoder/conv2d_2/kernel*(
_output_shapes
:*
dtype0

table_decoder/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nametable_decoder/conv2d_2/bias

/table_decoder/conv2d_2/bias/Read/ReadVariableOpReadVariableOptable_decoder/conv2d_2/bias*
_output_shapes	
:*
dtype0
³
'table_decoder/conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'table_decoder/conv2d_transpose_1/kernel
¬
;table_decoder/conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOp'table_decoder/conv2d_transpose_1/kernel*'
_output_shapes
:*
dtype0
¢
%table_decoder/conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%table_decoder/conv2d_transpose_1/bias

9table_decoder/conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOp%table_decoder/conv2d_transpose_1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ã
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer_with_weights-15
layer-20
layer-21
layer_with_weights-16
layer-22
layer-23
layer_with_weights-17
layer-24
layer-25
layer_with_weights-18
layer-26
layer_with_weights-19
layer-27
	optimizer
loss
regularization_losses
 	variables
!trainable_variables
"	keras_api
#
signatures
 
h

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
R
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
h

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
R
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
h

Pkernel
Qbias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
h

Vkernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
R
\regularization_losses
]	variables
^trainable_variables
_	keras_api
h

`kernel
abias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
h

fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
h

lkernel
mbias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
h

rkernel
sbias
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
R
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
j

|kernel
}bias
~regularization_losses
	variables
trainable_variables
	keras_api
n
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
n
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
n
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
n
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
 trainable_variables
¡	keras_api
n
¢kernel
	£bias
¤regularization_losses
¥	variables
¦trainable_variables
§	keras_api
V
¨regularization_losses
©	variables
ªtrainable_variables
«	keras_api
µ

¬conv1
­	upsample1
®	upsample2
¯	upsample3
°	upsample4
±convtraspose
²regularization_losses
³	variables
´trainable_variables
µ	keras_api
Í

¶conv1

·conv2

¸drop1
¹	upsample1
º	upsample2
»	upsample3
¼	upsample4
½convtraspose
¾regularization_losses
¿	variables
Àtrainable_variables
Á	keras_api
 
 
 
ú
$0
%1
*2
+3
44
55
:6
;7
D8
E9
J10
K11
P12
Q13
V14
W15
`16
a17
f18
g19
l20
m21
r22
s23
|24
}25
26
27
28
29
30
31
32
33
¢34
£35
Â36
Ã37
Ä38
Å39
Æ40
Ç41
È42
É43
Ê44
Ë45
t
0
1
¢2
£3
Â4
Ã5
Ä6
Å7
Æ8
Ç9
È10
É11
Ê12
Ë13
²
Ìlayers
regularization_losses
Ínon_trainable_variables
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
 	variables
!trainable_variables
 
_]
VARIABLE_VALUEblock1_conv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1
 
²
Ñlayers
 Òlayer_regularization_losses
Ónon_trainable_variables
Ômetrics
&regularization_losses
Õlayer_metrics
'	variables
(trainable_variables
_]
VARIABLE_VALUEblock1_conv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1
 
²
Ölayers
 ×layer_regularization_losses
Ønon_trainable_variables
Ùmetrics
,regularization_losses
Úlayer_metrics
-	variables
.trainable_variables
 
 
 
²
Ûlayers
 Ülayer_regularization_losses
Ýnon_trainable_variables
Þmetrics
0regularization_losses
ßlayer_metrics
1	variables
2trainable_variables
_]
VARIABLE_VALUEblock2_conv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2_conv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51
 
²
àlayers
 álayer_regularization_losses
ânon_trainable_variables
ãmetrics
6regularization_losses
älayer_metrics
7	variables
8trainable_variables
_]
VARIABLE_VALUEblock2_conv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2_conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1
 
²
ålayers
 ælayer_regularization_losses
çnon_trainable_variables
èmetrics
<regularization_losses
élayer_metrics
=	variables
>trainable_variables
 
 
 
²
êlayers
 ëlayer_regularization_losses
ìnon_trainable_variables
ímetrics
@regularization_losses
îlayer_metrics
A	variables
Btrainable_variables
_]
VARIABLE_VALUEblock3_conv1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1
 
²
ïlayers
 ðlayer_regularization_losses
ñnon_trainable_variables
òmetrics
Fregularization_losses
ólayer_metrics
G	variables
Htrainable_variables
_]
VARIABLE_VALUEblock3_conv2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1
 
²
ôlayers
 õlayer_regularization_losses
önon_trainable_variables
÷metrics
Lregularization_losses
ølayer_metrics
M	variables
Ntrainable_variables
_]
VARIABLE_VALUEblock3_conv3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1
 
²
ùlayers
 úlayer_regularization_losses
ûnon_trainable_variables
ümetrics
Rregularization_losses
ýlayer_metrics
S	variables
Ttrainable_variables
_]
VARIABLE_VALUEblock3_conv4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1
 
²
þlayers
 ÿlayer_regularization_losses
non_trainable_variables
metrics
Xregularization_losses
layer_metrics
Y	variables
Ztrainable_variables
 
 
 
²
layers
 layer_regularization_losses
non_trainable_variables
metrics
\regularization_losses
layer_metrics
]	variables
^trainable_variables
_]
VARIABLE_VALUEblock4_conv1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock4_conv1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

`0
a1
 
²
layers
 layer_regularization_losses
non_trainable_variables
metrics
bregularization_losses
layer_metrics
c	variables
dtrainable_variables
_]
VARIABLE_VALUEblock4_conv2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock4_conv2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

f0
g1
 
²
layers
 layer_regularization_losses
non_trainable_variables
metrics
hregularization_losses
layer_metrics
i	variables
jtrainable_variables
`^
VARIABLE_VALUEblock4_conv3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock4_conv3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1
 
²
layers
 layer_regularization_losses
non_trainable_variables
metrics
nregularization_losses
layer_metrics
o	variables
ptrainable_variables
`^
VARIABLE_VALUEblock4_conv4/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock4_conv4/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

r0
s1
 
²
layers
 layer_regularization_losses
non_trainable_variables
metrics
tregularization_losses
layer_metrics
u	variables
vtrainable_variables
 
 
 
²
layers
 layer_regularization_losses
non_trainable_variables
metrics
xregularization_losses
 layer_metrics
y	variables
ztrainable_variables
`^
VARIABLE_VALUEblock5_conv1/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv1/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

|0
}1
 
³
¡layers
 ¢layer_regularization_losses
£non_trainable_variables
¤metrics
~regularization_losses
¥layer_metrics
	variables
trainable_variables
`^
VARIABLE_VALUEblock5_conv2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv2/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
µ
¦layers
 §layer_regularization_losses
¨non_trainable_variables
©metrics
regularization_losses
ªlayer_metrics
	variables
trainable_variables
`^
VARIABLE_VALUEblock5_conv3/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv3/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
µ
«layers
 ¬layer_regularization_losses
­non_trainable_variables
®metrics
regularization_losses
¯layer_metrics
	variables
trainable_variables
`^
VARIABLE_VALUEblock5_conv4/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv4/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
µ
°layers
 ±layer_regularization_losses
²non_trainable_variables
³metrics
regularization_losses
´layer_metrics
	variables
trainable_variables
 
 
 
µ
µlayers
 ¶layer_regularization_losses
·non_trainable_variables
¸metrics
regularization_losses
¹layer_metrics
	variables
trainable_variables
b`
VARIABLE_VALUEblock_6_conv_1/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEblock_6_conv_1/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
ºlayers
 »layer_regularization_losses
¼non_trainable_variables
½metrics
regularization_losses
¾layer_metrics
	variables
trainable_variables
 
 
 
µ
¿layers
 Àlayer_regularization_losses
Ánon_trainable_variables
Âmetrics
regularization_losses
Ãlayer_metrics
	variables
 trainable_variables
b`
VARIABLE_VALUEblock_6_conv_2/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEblock_6_conv_2/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

¢0
£1

¢0
£1
µ
Älayers
 Ålayer_regularization_losses
Ænon_trainable_variables
Çmetrics
¤regularization_losses
Èlayer_metrics
¥	variables
¦trainable_variables
 
 
 
µ
Élayers
 Êlayer_regularization_losses
Ënon_trainable_variables
Ìmetrics
¨regularization_losses
Ílayer_metrics
©	variables
ªtrainable_variables
n
Âkernel
	Ãbias
Îregularization_losses
Ï	variables
Ðtrainable_variables
Ñ	keras_api
V
Òregularization_losses
Ó	variables
Ôtrainable_variables
Õ	keras_api
V
Öregularization_losses
×	variables
Øtrainable_variables
Ù	keras_api
V
Úregularization_losses
Û	variables
Ütrainable_variables
Ý	keras_api
V
Þregularization_losses
ß	variables
àtrainable_variables
á	keras_api
n
Äkernel
	Åbias
âregularization_losses
ã	variables
ätrainable_variables
å	keras_api
 
 
Â0
Ã1
Ä2
Å3
 
Â0
Ã1
Ä2
Å3
µ
ælayers
 çlayer_regularization_losses
ènon_trainable_variables
émetrics
²regularization_losses
êlayer_metrics
³	variables
´trainable_variables
n
Ækernel
	Çbias
ëregularization_losses
ì	variables
ítrainable_variables
î	keras_api
n
Èkernel
	Ébias
ïregularization_losses
ð	variables
ñtrainable_variables
ò	keras_api
V
óregularization_losses
ô	variables
õtrainable_variables
ö	keras_api
V
÷regularization_losses
ø	variables
ùtrainable_variables
ú	keras_api
V
ûregularization_losses
ü	variables
ýtrainable_variables
þ	keras_api
V
ÿregularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
n
Êkernel
	Ëbias
regularization_losses
	variables
trainable_variables
	keras_api
 
0
Æ0
Ç1
È2
É3
Ê4
Ë5
0
Æ0
Ç1
È2
É3
Ê4
Ë5
µ
layers
 layer_regularization_losses
non_trainable_variables
metrics
¾regularization_losses
layer_metrics
¿	variables
Àtrainable_variables
VT
VARIABLE_VALUEcol_decoder/conv2d/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcol_decoder/conv2d/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#col_decoder/conv2d_transpose/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!col_decoder/conv2d_transpose/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtable_decoder/conv2d_1/kernel'variables/40/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEtable_decoder/conv2d_1/bias'variables/41/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtable_decoder/conv2d_2/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEtable_decoder/conv2d_2/bias'variables/43/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'table_decoder/conv2d_transpose_1/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%table_decoder/conv2d_transpose_1/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE
Ö
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
ü
$0
%1
*2
+3
44
55
:6
;7
D8
E9
J10
K11
P12
Q13
V14
W15
`16
a17
f18
g19
l20
m21
r22
s23
|24
}25
26
27
28
29
30
31
 
 
 
 
 

$0
%1
 
 
 
 

*0
+1
 
 
 
 
 
 
 
 
 

40
51
 
 
 
 

:0
;1
 
 
 
 
 
 
 
 
 

D0
E1
 
 
 
 

J0
K1
 
 
 
 

P0
Q1
 
 
 
 

V0
W1
 
 
 
 
 
 
 
 
 

`0
a1
 
 
 
 

f0
g1
 
 
 
 

l0
m1
 
 
 
 

r0
s1
 
 
 
 
 
 
 
 
 

|0
}1
 
 
 
 

0
1
 
 
 
 

0
1
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Â0
Ã1

Â0
Ã1
µ
layers
 layer_regularization_losses
non_trainable_variables
metrics
Îregularization_losses
layer_metrics
Ï	variables
Ðtrainable_variables
 
 
 
µ
layers
 layer_regularization_losses
non_trainable_variables
metrics
Òregularization_losses
layer_metrics
Ó	variables
Ôtrainable_variables
 
 
 
µ
layers
 layer_regularization_losses
non_trainable_variables
metrics
Öregularization_losses
layer_metrics
×	variables
Øtrainable_variables
 
 
 
µ
layers
  layer_regularization_losses
¡non_trainable_variables
¢metrics
Úregularization_losses
£layer_metrics
Û	variables
Ütrainable_variables
 
 
 
µ
¤layers
 ¥layer_regularization_losses
¦non_trainable_variables
§metrics
Þregularization_losses
¨layer_metrics
ß	variables
àtrainable_variables
 

Ä0
Å1

Ä0
Å1
µ
©layers
 ªlayer_regularization_losses
«non_trainable_variables
¬metrics
âregularization_losses
­layer_metrics
ã	variables
ätrainable_variables
0
¬0
­1
®2
¯3
°4
±5
 
 
 
 
 

Æ0
Ç1

Æ0
Ç1
µ
®layers
 ¯layer_regularization_losses
°non_trainable_variables
±metrics
ëregularization_losses
²layer_metrics
ì	variables
ítrainable_variables
 

È0
É1

È0
É1
µ
³layers
 ´layer_regularization_losses
µnon_trainable_variables
¶metrics
ïregularization_losses
·layer_metrics
ð	variables
ñtrainable_variables
 
 
 
µ
¸layers
 ¹layer_regularization_losses
ºnon_trainable_variables
»metrics
óregularization_losses
¼layer_metrics
ô	variables
õtrainable_variables
 
 
 
µ
½layers
 ¾layer_regularization_losses
¿non_trainable_variables
Àmetrics
÷regularization_losses
Álayer_metrics
ø	variables
ùtrainable_variables
 
 
 
µ
Âlayers
 Ãlayer_regularization_losses
Änon_trainable_variables
Åmetrics
ûregularization_losses
Ælayer_metrics
ü	variables
ýtrainable_variables
 
 
 
µ
Çlayers
 Èlayer_regularization_losses
Énon_trainable_variables
Êmetrics
ÿregularization_losses
Ëlayer_metrics
	variables
trainable_variables
 
 
 
µ
Ìlayers
 Ílayer_regularization_losses
Înon_trainable_variables
Ïmetrics
regularization_losses
Ðlayer_metrics
	variables
trainable_variables
 

Ê0
Ë1

Ê0
Ë1
µ
Ñlayers
 Òlayer_regularization_losses
Ónon_trainable_variables
Ômetrics
regularization_losses
Õlayer_metrics
	variables
trainable_variables
@
¶0
·1
¸2
¹3
º4
»5
¼6
½7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_Input_LayerPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ  
²
StatefulPartitionedCallStatefulPartitionedCallserving_default_Input_Layerblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biasblock_6_conv_1/kernelblock_6_conv_1/biasblock_6_conv_2/kernelblock_6_conv_2/biastable_decoder/conv2d_1/kerneltable_decoder/conv2d_1/biastable_decoder/conv2d_2/kerneltable_decoder/conv2d_2/bias'table_decoder/conv2d_transpose_1/kernel%table_decoder/conv2d_transpose_1/biascol_decoder/conv2d/kernelcol_decoder/conv2d/bias#col_decoder/conv2d_transpose/kernel!col_decoder/conv2d_transpose/bias*:
Tin3
12/*
Tout
2*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  *P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.**
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference_signature_wrapper_3200
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ù
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block3_conv4/kernel/Read/ReadVariableOp%block3_conv4/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block4_conv4/kernel/Read/ReadVariableOp%block4_conv4/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOp'block5_conv4/kernel/Read/ReadVariableOp%block5_conv4/bias/Read/ReadVariableOp)block_6_conv_1/kernel/Read/ReadVariableOp'block_6_conv_1/bias/Read/ReadVariableOp)block_6_conv_2/kernel/Read/ReadVariableOp'block_6_conv_2/bias/Read/ReadVariableOp-col_decoder/conv2d/kernel/Read/ReadVariableOp+col_decoder/conv2d/bias/Read/ReadVariableOp7col_decoder/conv2d_transpose/kernel/Read/ReadVariableOp5col_decoder/conv2d_transpose/bias/Read/ReadVariableOp1table_decoder/conv2d_1/kernel/Read/ReadVariableOp/table_decoder/conv2d_1/bias/Read/ReadVariableOp1table_decoder/conv2d_2/kernel/Read/ReadVariableOp/table_decoder/conv2d_2/bias/Read/ReadVariableOp;table_decoder/conv2d_transpose_1/kernel/Read/ReadVariableOp9table_decoder/conv2d_transpose_1/bias/Read/ReadVariableOpConst*;
Tin4
220*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*&
f!R
__inference__traced_save_4578
¼

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biasblock_6_conv_1/kernelblock_6_conv_1/biasblock_6_conv_2/kernelblock_6_conv_2/biascol_decoder/conv2d/kernelcol_decoder/conv2d/bias#col_decoder/conv2d_transpose/kernel!col_decoder/conv2d_transpose/biastable_decoder/conv2d_1/kerneltable_decoder/conv2d_1/biastable_decoder/conv2d_2/kerneltable_decoder/conv2d_2/bias'table_decoder/conv2d_transpose_1/kernel%table_decoder/conv2d_transpose_1/bias*:
Tin3
12/*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_restore_4728Ë
ñf
®
__inference__traced_save_4578
file_prefix2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block3_conv4_kernel_read_readvariableop0
,savev2_block3_conv4_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block4_conv4_kernel_read_readvariableop0
,savev2_block4_conv4_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop2
.savev2_block5_conv4_kernel_read_readvariableop0
,savev2_block5_conv4_bias_read_readvariableop4
0savev2_block_6_conv_1_kernel_read_readvariableop2
.savev2_block_6_conv_1_bias_read_readvariableop4
0savev2_block_6_conv_2_kernel_read_readvariableop2
.savev2_block_6_conv_2_bias_read_readvariableop8
4savev2_col_decoder_conv2d_kernel_read_readvariableop6
2savev2_col_decoder_conv2d_bias_read_readvariableopB
>savev2_col_decoder_conv2d_transpose_kernel_read_readvariableop@
<savev2_col_decoder_conv2d_transpose_bias_read_readvariableop<
8savev2_table_decoder_conv2d_1_kernel_read_readvariableop:
6savev2_table_decoder_conv2d_1_bias_read_readvariableop<
8savev2_table_decoder_conv2d_2_kernel_read_readvariableop:
6savev2_table_decoder_conv2d_2_bias_read_readvariableopF
Bsavev2_table_decoder_conv2d_transpose_1_kernel_read_readvariableopD
@savev2_table_decoder_conv2d_transpose_1_bias_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_277acd1f34a247709606346c0162c81f/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameé
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*û
valueñBî.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesä
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÏ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block3_conv4_kernel_read_readvariableop,savev2_block3_conv4_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block4_conv4_kernel_read_readvariableop,savev2_block4_conv4_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop.savev2_block5_conv4_kernel_read_readvariableop,savev2_block5_conv4_bias_read_readvariableop0savev2_block_6_conv_1_kernel_read_readvariableop.savev2_block_6_conv_1_bias_read_readvariableop0savev2_block_6_conv_2_kernel_read_readvariableop.savev2_block_6_conv_2_bias_read_readvariableop4savev2_col_decoder_conv2d_kernel_read_readvariableop2savev2_col_decoder_conv2d_bias_read_readvariableop>savev2_col_decoder_conv2d_transpose_kernel_read_readvariableop<savev2_col_decoder_conv2d_transpose_bias_read_readvariableop8savev2_table_decoder_conv2d_1_kernel_read_readvariableop6savev2_table_decoder_conv2d_1_bias_read_readvariableop8savev2_table_decoder_conv2d_2_kernel_read_readvariableop6savev2_table_decoder_conv2d_2_bias_read_readvariableopBsavev2_table_decoder_conv2d_transpose_1_kernel_read_readvariableop@savev2_table_decoder_conv2d_transpose_1_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesÏ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ã
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*û
_input_shapesé
æ: :@:@:@@:@:@:::::::::::::::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::! 

_output_shapes	
::.!*
(
_output_shapes
::!"

_output_shapes	
::.#*
(
_output_shapes
::!$

_output_shapes	
::.%*
(
_output_shapes
::!&

_output_shapes	
::-')
'
_output_shapes
:: (

_output_shapes
::.)*
(
_output_shapes
::!*

_output_shapes	
::.+*
(
_output_shapes
::!,

_output_shapes	
::--)
'
_output_shapes
:: .

_output_shapes
::/

_output_shapes
: 
ç

+__inference_block5_conv1_layer_call_fn_1568

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_15582
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

J
.__inference_up_sampling2d_3_layer_call_fn_1788

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_17822
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç

+__inference_block4_conv3_layer_call_fn_1512

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_15022
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
½

®
F__inference_block2_conv2_layer_call_and_return_conditional_losses_1324

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
½

®
F__inference_block3_conv1_layer_call_and_return_conditional_losses_1358

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
û
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_1440

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ès
Ë
E__inference_col_decoder_layer_call_and_return_conditional_losses_4166
input_0
input_1
input_2)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource
identity¬
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp»
conv2d/Conv2DConv2Dinput_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d/Conv2D¢
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¥
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/BiasAddv
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/Relus
up_sampling2d/ShapeShapeconv2d/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2¢
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mulã
#up_sampling2d/resize/ResizeBilinearResizeBilinearconv2d/Relu:activations:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
half_pixel_centers(2%
#up_sampling2d/resize/ResizeBilineart
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisÙ
concatenate/concatConcatV2input_14up_sampling2d/resize/ResizeBilinear:resized_images:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
concatenate/concaty
up_sampling2d_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2®
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mulë
%up_sampling2d_1/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
half_pixel_centers(2'
%up_sampling2d_1/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisá
concatenate_1/concatConcatV2input_26up_sampling2d_1/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
concatenate_1/concat{
up_sampling2d_2/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shape
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stack
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2®
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_2/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor
up_sampling2d_3/ShapeShape=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shape
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stack
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2®
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul¤
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_3/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor
conv2d_transpose/ShapeShape=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2È
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slice
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_1/stack
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2Ò
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1
&conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_2/stack
(conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_1
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_2Ò
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_2r
conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul/y 
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mulv
conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul_1/y¦
conv2d_transpose/mul_1Mul)conv2d_transpose/strided_slice_2:output:0!conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mul_1v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3è
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/mul:z:0conv2d_transpose/mul_1:z:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack
&conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_3/stack
(conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_1
(conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_2Ò
 conv2d_transpose/strided_slice_3StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_3/stack:output:01conv2d_transpose/strided_slice_3/stack_1:output:01conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_3ç
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOpÛ
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose¿
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOpØ
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose/BiasAdd
&conv2d_transpose/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&conv2d_transpose/Max/reduction_indicesÔ
conv2d_transpose/MaxMax!conv2d_transpose/BiasAdd:output:0/conv2d_transpose/Max/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2
conv2d_transpose/Max±
conv2d_transpose/subSub!conv2d_transpose/BiasAdd:output:0conv2d_transpose/Max:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose/sub
conv2d_transpose/ExpExpconv2d_transpose/sub:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose/Exp
&conv2d_transpose/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&conv2d_transpose/Sum/reduction_indicesË
conv2d_transpose/SumSumconv2d_transpose/Exp:y:0/conv2d_transpose/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2
conv2d_transpose/Sum´
conv2d_transpose/truedivRealDivconv2d_transpose/Exp:y:0conv2d_transpose/Sum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose/truedivz
IdentityIdentityconv2d_transpose/truediv:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22:ÿÿÿÿÿÿÿÿÿdd:::::Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/0:YU
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input/1:YU
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!
_user_specified_name	input/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

e
I__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_1953

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å

+__inference_block2_conv1_layer_call_fn_1312

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_13022
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
É
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_4066

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÇ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸

¨
@__inference_conv2d_layer_call_and_return_conditional_losses_1702

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp·
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
º
Ø
,__inference_table_decoder_layer_call_fn_4393
input_0
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_table_decoder_layer_call_and_return_conditional_losses_23622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*
_input_shapesn
l:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22:ÿÿÿÿÿÿÿÿÿdd::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/0:YU
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input/1:YU
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!
_user_specified_name	input/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ç

+__inference_block5_conv3_layer_call_fn_1612

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_16022
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
½

®
F__inference_block3_conv3_layer_call_and_return_conditional_losses_1402

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ô)
¾
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2000

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ì
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3´
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpð
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Max/reduction_indices 
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Max}
subSubBiasAdd:output:0Max:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
subf
ExpExpsub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indices
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Sum
truedivRealDivExp:y:0Sum:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
truedivy
IdentityIdentitytruediv:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ÿ
F
*__inference_block2_pool_layer_call_fn_1346

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_13402
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

+__inference_block1_conv2_layer_call_fn_1278

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_12682
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ú
z
%__inference_conv2d_layer_call_fn_1712

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_17022
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

e
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_1782

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
F
*__inference_block1_pool_layer_call_fn_1290

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_12842
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

_
&__inference_dropout_layer_call_fn_4049

inputs
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_21162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
ë
?__inference_model_layer_call_and_return_conditional_losses_3829

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block3_conv3_conv2d_readvariableop_resource0
,block3_conv3_biasadd_readvariableop_resource/
+block3_conv4_conv2d_readvariableop_resource0
,block3_conv4_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block4_conv3_conv2d_readvariableop_resource0
,block4_conv3_biasadd_readvariableop_resource/
+block4_conv4_conv2d_readvariableop_resource0
,block4_conv4_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource/
+block5_conv3_conv2d_readvariableop_resource0
,block5_conv3_biasadd_readvariableop_resource/
+block5_conv4_conv2d_readvariableop_resource0
,block5_conv4_biasadd_readvariableop_resource1
-block_6_conv_1_conv2d_readvariableop_resource2
.block_6_conv_1_biasadd_readvariableop_resource1
-block_6_conv_2_conv2d_readvariableop_resource2
.block_6_conv_2_biasadd_readvariableop_resource9
5table_decoder_conv2d_1_conv2d_readvariableop_resource:
6table_decoder_conv2d_1_biasadd_readvariableop_resource9
5table_decoder_conv2d_2_conv2d_readvariableop_resource:
6table_decoder_conv2d_2_biasadd_readvariableop_resourceM
Itable_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceD
@table_decoder_conv2d_transpose_1_biasadd_readvariableop_resource5
1col_decoder_conv2d_conv2d_readvariableop_resource6
2col_decoder_conv2d_biasadd_readvariableop_resourceI
Ecol_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource@
<col_decoder_conv2d_transpose_biasadd_readvariableop_resource
identity

identity_1¼
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOpÌ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingSAME*
strides
2
block1_conv1/Conv2D³
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp¾
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
block1_conv1/BiasAdd
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
block1_conv1/Relu¼
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOpå
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingSAME*
strides
2
block1_conv2/Conv2D³
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp¾
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
block1_conv2/BiasAdd
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
block1_conv2/ReluÅ
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool½
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"block2_conv1/Conv2D/ReadVariableOpã
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block2_conv1/Conv2D´
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp¿
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2
block2_conv1/BiasAdd
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2
block2_conv1/Relu¾
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block2_conv2/Conv2D/ReadVariableOpæ
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block2_conv2/Conv2D´
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp¿
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2
block2_conv2/BiasAdd
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2
block2_conv2/ReluÆ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool¾
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv1/Conv2D/ReadVariableOpã
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
paddingSAME*
strides
2
block3_conv1/Conv2D´
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp¿
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv1/BiasAdd
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv1/Relu¾
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv2/Conv2D/ReadVariableOpæ
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
paddingSAME*
strides
2
block3_conv2/Conv2D´
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp¿
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv2/BiasAdd
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv2/Relu¾
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv3/Conv2D/ReadVariableOpæ
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
paddingSAME*
strides
2
block3_conv3/Conv2D´
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp¿
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv3/BiasAdd
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv3/Relu¾
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv4/Conv2D/ReadVariableOpæ
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
paddingSAME*
strides
2
block3_conv4/Conv2D´
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv4/BiasAdd/ReadVariableOp¿
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv4/BiasAdd
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv4/ReluÄ
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool¾
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv1/Conv2D/ReadVariableOpá
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
2
block4_conv1/Conv2D´
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp½
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv1/BiasAdd
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv1/Relu¾
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv2/Conv2D/ReadVariableOpä
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
2
block4_conv2/Conv2D´
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp½
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv2/BiasAdd
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv2/Relu¾
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv3/Conv2D/ReadVariableOpä
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
2
block4_conv3/Conv2D´
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp½
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv3/BiasAdd
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv3/Relu¾
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv4/Conv2D/ReadVariableOpä
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
2
block4_conv4/Conv2D´
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv4/BiasAdd/ReadVariableOp½
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv4/BiasAdd
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv4/ReluÄ
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool¾
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5_conv1/Conv2D/ReadVariableOpá
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
block5_conv1/Conv2D´
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp½
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv1/BiasAdd
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv1/Relu¾
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5_conv2/Conv2D/ReadVariableOpä
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
block5_conv2/Conv2D´
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOp½
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv2/BiasAdd
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv2/Relu¾
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5_conv3/Conv2D/ReadVariableOpä
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
block5_conv3/Conv2D´
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOp½
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv3/BiasAdd
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv3/Relu¾
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5_conv4/Conv2D/ReadVariableOpä
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
block5_conv4/Conv2D´
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5_conv4/BiasAdd/ReadVariableOp½
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv4/BiasAdd
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv4/ReluÄ
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPoolÄ
$block_6_conv_1/Conv2D/ReadVariableOpReadVariableOp-block_6_conv_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02&
$block_6_conv_1/Conv2D/ReadVariableOpè
block_6_conv_1/Conv2DConv2Dblock5_pool/MaxPool:output:0,block_6_conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
block_6_conv_1/Conv2Dº
%block_6_conv_1/BiasAdd/ReadVariableOpReadVariableOp.block_6_conv_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%block_6_conv_1/BiasAdd/ReadVariableOpÅ
block_6_conv_1/BiasAddBiasAddblock_6_conv_1/Conv2D:output:0-block_6_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block_6_conv_1/BiasAdd
block_6_conv_1/ReluRelublock_6_conv_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block_6_conv_1/Relu
dropout/IdentityIdentity!block_6_conv_1/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/IdentityÄ
$block_6_conv_2/Conv2D/ReadVariableOpReadVariableOp-block_6_conv_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02&
$block_6_conv_2/Conv2D/ReadVariableOpå
block_6_conv_2/Conv2DConv2Ddropout/Identity:output:0,block_6_conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
block_6_conv_2/Conv2Dº
%block_6_conv_2/BiasAdd/ReadVariableOpReadVariableOp.block_6_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%block_6_conv_2/BiasAdd/ReadVariableOpÅ
block_6_conv_2/BiasAddBiasAddblock_6_conv_2/Conv2D:output:0-block_6_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block_6_conv_2/BiasAdd
block_6_conv_2/ReluRelublock_6_conv_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block_6_conv_2/Relu
dropout_1/IdentityIdentity!block_6_conv_2/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/IdentityÜ
,table_decoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5table_decoder_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,table_decoder/conv2d_1/Conv2D/ReadVariableOpÿ
table_decoder/conv2d_1/Conv2DConv2Ddropout_1/Identity:output:04table_decoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
table_decoder/conv2d_1/Conv2DÒ
-table_decoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6table_decoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-table_decoder/conv2d_1/BiasAdd/ReadVariableOpå
table_decoder/conv2d_1/BiasAddBiasAdd&table_decoder/conv2d_1/Conv2D:output:05table_decoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
table_decoder/conv2d_1/BiasAdd¦
table_decoder/conv2d_1/ReluRelu'table_decoder/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
table_decoder/conv2d_1/Relu¶
 table_decoder/dropout_2/IdentityIdentity)table_decoder/conv2d_1/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 table_decoder/dropout_2/IdentityÜ
,table_decoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5table_decoder_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,table_decoder/conv2d_2/Conv2D/ReadVariableOp
table_decoder/conv2d_2/Conv2DConv2D)table_decoder/dropout_2/Identity:output:04table_decoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
table_decoder/conv2d_2/Conv2DÒ
-table_decoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6table_decoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-table_decoder/conv2d_2/BiasAdd/ReadVariableOpå
table_decoder/conv2d_2/BiasAddBiasAdd&table_decoder/conv2d_2/Conv2D:output:05table_decoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
table_decoder/conv2d_2/BiasAdd¦
table_decoder/conv2d_2/ReluRelu'table_decoder/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
table_decoder/conv2d_2/Relu£
#table_decoder/up_sampling2d_4/ShapeShape)table_decoder/conv2d_1/Relu:activations:0*
T0*
_output_shapes
:2%
#table_decoder/up_sampling2d_4/Shape°
1table_decoder/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1table_decoder/up_sampling2d_4/strided_slice/stack´
3table_decoder/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_4/strided_slice/stack_1´
3table_decoder/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_4/strided_slice/stack_2
+table_decoder/up_sampling2d_4/strided_sliceStridedSlice,table_decoder/up_sampling2d_4/Shape:output:0:table_decoder/up_sampling2d_4/strided_slice/stack:output:0<table_decoder/up_sampling2d_4/strided_slice/stack_1:output:0<table_decoder/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2-
+table_decoder/up_sampling2d_4/strided_slice
#table_decoder/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#table_decoder/up_sampling2d_4/ConstÖ
!table_decoder/up_sampling2d_4/mulMul4table_decoder/up_sampling2d_4/strided_slice:output:0,table_decoder/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_4/mul£
3table_decoder/up_sampling2d_4/resize/ResizeBilinearResizeBilinear)table_decoder/conv2d_1/Relu:activations:0%table_decoder/up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
half_pixel_centers(25
3table_decoder/up_sampling2d_4/resize/ResizeBilinear
%table_decoder/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%table_decoder/concatenate/concat/axis¨
 table_decoder/concatenate/concatConcatV2block4_pool/MaxPool:output:0Dtable_decoder/up_sampling2d_4/resize/ResizeBilinear:resized_images:0.table_decoder/concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222"
 table_decoder/concatenate/concat£
#table_decoder/up_sampling2d_5/ShapeShape)table_decoder/concatenate/concat:output:0*
T0*
_output_shapes
:2%
#table_decoder/up_sampling2d_5/Shape°
1table_decoder/up_sampling2d_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1table_decoder/up_sampling2d_5/strided_slice/stack´
3table_decoder/up_sampling2d_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_5/strided_slice/stack_1´
3table_decoder/up_sampling2d_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_5/strided_slice/stack_2
+table_decoder/up_sampling2d_5/strided_sliceStridedSlice,table_decoder/up_sampling2d_5/Shape:output:0:table_decoder/up_sampling2d_5/strided_slice/stack:output:0<table_decoder/up_sampling2d_5/strided_slice/stack_1:output:0<table_decoder/up_sampling2d_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2-
+table_decoder/up_sampling2d_5/strided_slice
#table_decoder/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#table_decoder/up_sampling2d_5/ConstÖ
!table_decoder/up_sampling2d_5/mulMul4table_decoder/up_sampling2d_5/strided_slice:output:0,table_decoder/up_sampling2d_5/Const:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_5/mul£
3table_decoder/up_sampling2d_5/resize/ResizeBilinearResizeBilinear)table_decoder/concatenate/concat:output:0%table_decoder/up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
half_pixel_centers(25
3table_decoder/up_sampling2d_5/resize/ResizeBilinear
'table_decoder/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2)
'table_decoder/concatenate_1/concat/axis®
"table_decoder/concatenate_1/concatConcatV2block3_pool/MaxPool:output:0Dtable_decoder/up_sampling2d_5/resize/ResizeBilinear:resized_images:00table_decoder/concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2$
"table_decoder/concatenate_1/concat¥
#table_decoder/up_sampling2d_6/ShapeShape+table_decoder/concatenate_1/concat:output:0*
T0*
_output_shapes
:2%
#table_decoder/up_sampling2d_6/Shape°
1table_decoder/up_sampling2d_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1table_decoder/up_sampling2d_6/strided_slice/stack´
3table_decoder/up_sampling2d_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_6/strided_slice/stack_1´
3table_decoder/up_sampling2d_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_6/strided_slice/stack_2
+table_decoder/up_sampling2d_6/strided_sliceStridedSlice,table_decoder/up_sampling2d_6/Shape:output:0:table_decoder/up_sampling2d_6/strided_slice/stack:output:0<table_decoder/up_sampling2d_6/strided_slice/stack_1:output:0<table_decoder/up_sampling2d_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2-
+table_decoder/up_sampling2d_6/strided_slice
#table_decoder/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#table_decoder/up_sampling2d_6/ConstÖ
!table_decoder/up_sampling2d_6/mulMul4table_decoder/up_sampling2d_6/strided_slice:output:0,table_decoder/up_sampling2d_6/Const:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_6/mul¼
:table_decoder/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor+table_decoder/concatenate_1/concat:output:0%table_decoder/up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
half_pixel_centers(2<
:table_decoder/up_sampling2d_6/resize/ResizeNearestNeighborÅ
#table_decoder/up_sampling2d_7/ShapeShapeKtable_decoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2%
#table_decoder/up_sampling2d_7/Shape°
1table_decoder/up_sampling2d_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1table_decoder/up_sampling2d_7/strided_slice/stack´
3table_decoder/up_sampling2d_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_7/strided_slice/stack_1´
3table_decoder/up_sampling2d_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_7/strided_slice/stack_2
+table_decoder/up_sampling2d_7/strided_sliceStridedSlice,table_decoder/up_sampling2d_7/Shape:output:0:table_decoder/up_sampling2d_7/strided_slice/stack:output:0<table_decoder/up_sampling2d_7/strided_slice/stack_1:output:0<table_decoder/up_sampling2d_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2-
+table_decoder/up_sampling2d_7/strided_slice
#table_decoder/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#table_decoder/up_sampling2d_7/ConstÖ
!table_decoder/up_sampling2d_7/mulMul4table_decoder/up_sampling2d_7/strided_slice:output:0,table_decoder/up_sampling2d_7/Const:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_7/mulÜ
:table_decoder/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborKtable_decoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0%table_decoder/up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2<
:table_decoder/up_sampling2d_7/resize/ResizeNearestNeighborË
&table_decoder/conv2d_transpose_1/ShapeShapeKtable_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2(
&table_decoder/conv2d_transpose_1/Shape¶
4table_decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4table_decoder/conv2d_transpose_1/strided_slice/stackº
6table_decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6table_decoder/conv2d_transpose_1/strided_slice/stack_1º
6table_decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6table_decoder/conv2d_transpose_1/strided_slice/stack_2¨
.table_decoder/conv2d_transpose_1/strided_sliceStridedSlice/table_decoder/conv2d_transpose_1/Shape:output:0=table_decoder/conv2d_transpose_1/strided_slice/stack:output:0?table_decoder/conv2d_transpose_1/strided_slice/stack_1:output:0?table_decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.table_decoder/conv2d_transpose_1/strided_sliceº
6table_decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6table_decoder/conv2d_transpose_1/strided_slice_1/stack¾
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_1¾
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_2²
0table_decoder/conv2d_transpose_1/strided_slice_1StridedSlice/table_decoder/conv2d_transpose_1/Shape:output:0?table_decoder/conv2d_transpose_1/strided_slice_1/stack:output:0Atable_decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0Atable_decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0table_decoder/conv2d_transpose_1/strided_slice_1º
6table_decoder/conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6table_decoder/conv2d_transpose_1/strided_slice_2/stack¾
8table_decoder/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_2/stack_1¾
8table_decoder/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_2/stack_2²
0table_decoder/conv2d_transpose_1/strided_slice_2StridedSlice/table_decoder/conv2d_transpose_1/Shape:output:0?table_decoder/conv2d_transpose_1/strided_slice_2/stack:output:0Atable_decoder/conv2d_transpose_1/strided_slice_2/stack_1:output:0Atable_decoder/conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0table_decoder/conv2d_transpose_1/strided_slice_2
&table_decoder/conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&table_decoder/conv2d_transpose_1/mul/yà
$table_decoder/conv2d_transpose_1/mulMul9table_decoder/conv2d_transpose_1/strided_slice_1:output:0/table_decoder/conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2&
$table_decoder/conv2d_transpose_1/mul
(table_decoder/conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(table_decoder/conv2d_transpose_1/mul_1/yæ
&table_decoder/conv2d_transpose_1/mul_1Mul9table_decoder/conv2d_transpose_1/strided_slice_2:output:01table_decoder/conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2(
&table_decoder/conv2d_transpose_1/mul_1
(table_decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(table_decoder/conv2d_transpose_1/stack/3È
&table_decoder/conv2d_transpose_1/stackPack7table_decoder/conv2d_transpose_1/strided_slice:output:0(table_decoder/conv2d_transpose_1/mul:z:0*table_decoder/conv2d_transpose_1/mul_1:z:01table_decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&table_decoder/conv2d_transpose_1/stackº
6table_decoder/conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6table_decoder/conv2d_transpose_1/strided_slice_3/stack¾
8table_decoder/conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_3/stack_1¾
8table_decoder/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_3/stack_2²
0table_decoder/conv2d_transpose_1/strided_slice_3StridedSlice/table_decoder/conv2d_transpose_1/stack:output:0?table_decoder/conv2d_transpose_1/strided_slice_3/stack:output:0Atable_decoder/conv2d_transpose_1/strided_slice_3/stack_1:output:0Atable_decoder/conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0table_decoder/conv2d_transpose_1/strided_slice_3
@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpItable_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype02B
@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp©
1table_decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput/table_decoder/conv2d_transpose_1/stack:output:0Htable_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Ktable_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
23
1table_decoder/conv2d_transpose_1/conv2d_transposeï
7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp@table_decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp
(table_decoder/conv2d_transpose_1/BiasAddBiasAdd:table_decoder/conv2d_transpose_1/conv2d_transpose:output:0?table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2*
(table_decoder/conv2d_transpose_1/BiasAdd»
6table_decoder/conv2d_transpose_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ28
6table_decoder/conv2d_transpose_1/Max/reduction_indices
$table_decoder/conv2d_transpose_1/MaxMax1table_decoder/conv2d_transpose_1/BiasAdd:output:0?table_decoder/conv2d_transpose_1/Max/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2&
$table_decoder/conv2d_transpose_1/Maxñ
$table_decoder/conv2d_transpose_1/subSub1table_decoder/conv2d_transpose_1/BiasAdd:output:0-table_decoder/conv2d_transpose_1/Max:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2&
$table_decoder/conv2d_transpose_1/sub¹
$table_decoder/conv2d_transpose_1/ExpExp(table_decoder/conv2d_transpose_1/sub:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2&
$table_decoder/conv2d_transpose_1/Exp»
6table_decoder/conv2d_transpose_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ28
6table_decoder/conv2d_transpose_1/Sum/reduction_indices
$table_decoder/conv2d_transpose_1/SumSum(table_decoder/conv2d_transpose_1/Exp:y:0?table_decoder/conv2d_transpose_1/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2&
$table_decoder/conv2d_transpose_1/Sumô
(table_decoder/conv2d_transpose_1/truedivRealDiv(table_decoder/conv2d_transpose_1/Exp:y:0-table_decoder/conv2d_transpose_1/Sum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2*
(table_decoder/conv2d_transpose_1/truedivÐ
(col_decoder/conv2d/Conv2D/ReadVariableOpReadVariableOp1col_decoder_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(col_decoder/conv2d/Conv2D/ReadVariableOpó
col_decoder/conv2d/Conv2DConv2Ddropout_1/Identity:output:00col_decoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
col_decoder/conv2d/Conv2DÆ
)col_decoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp2col_decoder_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)col_decoder/conv2d/BiasAdd/ReadVariableOpÕ
col_decoder/conv2d/BiasAddBiasAdd"col_decoder/conv2d/Conv2D:output:01col_decoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
col_decoder/conv2d/BiasAdd
col_decoder/conv2d/ReluRelu#col_decoder/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
col_decoder/conv2d/Relu
col_decoder/up_sampling2d/ShapeShape%col_decoder/conv2d/Relu:activations:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d/Shape¨
-col_decoder/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-col_decoder/up_sampling2d/strided_slice/stack¬
/col_decoder/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/col_decoder/up_sampling2d/strided_slice/stack_1¬
/col_decoder/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/col_decoder/up_sampling2d/strided_slice/stack_2ê
'col_decoder/up_sampling2d/strided_sliceStridedSlice(col_decoder/up_sampling2d/Shape:output:06col_decoder/up_sampling2d/strided_slice/stack:output:08col_decoder/up_sampling2d/strided_slice/stack_1:output:08col_decoder/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'col_decoder/up_sampling2d/strided_slice
col_decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2!
col_decoder/up_sampling2d/ConstÆ
col_decoder/up_sampling2d/mulMul0col_decoder/up_sampling2d/strided_slice:output:0(col_decoder/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
col_decoder/up_sampling2d/mul
/col_decoder/up_sampling2d/resize/ResizeBilinearResizeBilinear%col_decoder/conv2d/Relu:activations:0!col_decoder/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
half_pixel_centers(21
/col_decoder/up_sampling2d/resize/ResizeBilinear
%col_decoder/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%col_decoder/concatenate_2/concat/axis¤
 col_decoder/concatenate_2/concatConcatV2block4_pool/MaxPool:output:0@col_decoder/up_sampling2d/resize/ResizeBilinear:resized_images:0.col_decoder/concatenate_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222"
 col_decoder/concatenate_2/concat
!col_decoder/up_sampling2d_1/ShapeShape)col_decoder/concatenate_2/concat:output:0*
T0*
_output_shapes
:2#
!col_decoder/up_sampling2d_1/Shape¬
/col_decoder/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/col_decoder/up_sampling2d_1/strided_slice/stack°
1col_decoder/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1col_decoder/up_sampling2d_1/strided_slice/stack_1°
1col_decoder/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1col_decoder/up_sampling2d_1/strided_slice/stack_2ö
)col_decoder/up_sampling2d_1/strided_sliceStridedSlice*col_decoder/up_sampling2d_1/Shape:output:08col_decoder/up_sampling2d_1/strided_slice/stack:output:0:col_decoder/up_sampling2d_1/strided_slice/stack_1:output:0:col_decoder/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)col_decoder/up_sampling2d_1/strided_slice
!col_decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2#
!col_decoder/up_sampling2d_1/ConstÎ
col_decoder/up_sampling2d_1/mulMul2col_decoder/up_sampling2d_1/strided_slice:output:0*col_decoder/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d_1/mul
1col_decoder/up_sampling2d_1/resize/ResizeBilinearResizeBilinear)col_decoder/concatenate_2/concat:output:0#col_decoder/up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
half_pixel_centers(23
1col_decoder/up_sampling2d_1/resize/ResizeBilinear
%col_decoder/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%col_decoder/concatenate_3/concat/axis¦
 col_decoder/concatenate_3/concatConcatV2block3_pool/MaxPool:output:0Bcol_decoder/up_sampling2d_1/resize/ResizeBilinear:resized_images:0.col_decoder/concatenate_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2"
 col_decoder/concatenate_3/concat
!col_decoder/up_sampling2d_2/ShapeShape)col_decoder/concatenate_3/concat:output:0*
T0*
_output_shapes
:2#
!col_decoder/up_sampling2d_2/Shape¬
/col_decoder/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/col_decoder/up_sampling2d_2/strided_slice/stack°
1col_decoder/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1col_decoder/up_sampling2d_2/strided_slice/stack_1°
1col_decoder/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1col_decoder/up_sampling2d_2/strided_slice/stack_2ö
)col_decoder/up_sampling2d_2/strided_sliceStridedSlice*col_decoder/up_sampling2d_2/Shape:output:08col_decoder/up_sampling2d_2/strided_slice/stack:output:0:col_decoder/up_sampling2d_2/strided_slice/stack_1:output:0:col_decoder/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)col_decoder/up_sampling2d_2/strided_slice
!col_decoder/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2#
!col_decoder/up_sampling2d_2/ConstÎ
col_decoder/up_sampling2d_2/mulMul2col_decoder/up_sampling2d_2/strided_slice:output:0*col_decoder/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d_2/mul´
8col_decoder/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor)col_decoder/concatenate_3/concat:output:0#col_decoder/up_sampling2d_2/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
half_pixel_centers(2:
8col_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor¿
!col_decoder/up_sampling2d_3/ShapeShapeIcol_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2#
!col_decoder/up_sampling2d_3/Shape¬
/col_decoder/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/col_decoder/up_sampling2d_3/strided_slice/stack°
1col_decoder/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1col_decoder/up_sampling2d_3/strided_slice/stack_1°
1col_decoder/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1col_decoder/up_sampling2d_3/strided_slice/stack_2ö
)col_decoder/up_sampling2d_3/strided_sliceStridedSlice*col_decoder/up_sampling2d_3/Shape:output:08col_decoder/up_sampling2d_3/strided_slice/stack:output:0:col_decoder/up_sampling2d_3/strided_slice/stack_1:output:0:col_decoder/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)col_decoder/up_sampling2d_3/strided_slice
!col_decoder/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2#
!col_decoder/up_sampling2d_3/ConstÎ
col_decoder/up_sampling2d_3/mulMul2col_decoder/up_sampling2d_3/strided_slice:output:0*col_decoder/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d_3/mulÔ
8col_decoder/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborIcol_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0#col_decoder/up_sampling2d_3/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2:
8col_decoder/up_sampling2d_3/resize/ResizeNearestNeighborÁ
"col_decoder/conv2d_transpose/ShapeShapeIcol_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2$
"col_decoder/conv2d_transpose/Shape®
0col_decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0col_decoder/conv2d_transpose/strided_slice/stack²
2col_decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2col_decoder/conv2d_transpose/strided_slice/stack_1²
2col_decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2col_decoder/conv2d_transpose/strided_slice/stack_2
*col_decoder/conv2d_transpose/strided_sliceStridedSlice+col_decoder/conv2d_transpose/Shape:output:09col_decoder/conv2d_transpose/strided_slice/stack:output:0;col_decoder/conv2d_transpose/strided_slice/stack_1:output:0;col_decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*col_decoder/conv2d_transpose/strided_slice²
2col_decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2col_decoder/conv2d_transpose/strided_slice_1/stack¶
4col_decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_1/stack_1¶
4col_decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_1/stack_2
,col_decoder/conv2d_transpose/strided_slice_1StridedSlice+col_decoder/conv2d_transpose/Shape:output:0;col_decoder/conv2d_transpose/strided_slice_1/stack:output:0=col_decoder/conv2d_transpose/strided_slice_1/stack_1:output:0=col_decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,col_decoder/conv2d_transpose/strided_slice_1²
2col_decoder/conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2col_decoder/conv2d_transpose/strided_slice_2/stack¶
4col_decoder/conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_2/stack_1¶
4col_decoder/conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_2/stack_2
,col_decoder/conv2d_transpose/strided_slice_2StridedSlice+col_decoder/conv2d_transpose/Shape:output:0;col_decoder/conv2d_transpose/strided_slice_2/stack:output:0=col_decoder/conv2d_transpose/strided_slice_2/stack_1:output:0=col_decoder/conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,col_decoder/conv2d_transpose/strided_slice_2
"col_decoder/conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"col_decoder/conv2d_transpose/mul/yÐ
 col_decoder/conv2d_transpose/mulMul5col_decoder/conv2d_transpose/strided_slice_1:output:0+col_decoder/conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2"
 col_decoder/conv2d_transpose/mul
$col_decoder/conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$col_decoder/conv2d_transpose/mul_1/yÖ
"col_decoder/conv2d_transpose/mul_1Mul5col_decoder/conv2d_transpose/strided_slice_2:output:0-col_decoder/conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2$
"col_decoder/conv2d_transpose/mul_1
$col_decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$col_decoder/conv2d_transpose/stack/3°
"col_decoder/conv2d_transpose/stackPack3col_decoder/conv2d_transpose/strided_slice:output:0$col_decoder/conv2d_transpose/mul:z:0&col_decoder/conv2d_transpose/mul_1:z:0-col_decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"col_decoder/conv2d_transpose/stack²
2col_decoder/conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2col_decoder/conv2d_transpose/strided_slice_3/stack¶
4col_decoder/conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_3/stack_1¶
4col_decoder/conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_3/stack_2
,col_decoder/conv2d_transpose/strided_slice_3StridedSlice+col_decoder/conv2d_transpose/stack:output:0;col_decoder/conv2d_transpose/strided_slice_3/stack:output:0=col_decoder/conv2d_transpose/strided_slice_3/stack_1:output:0=col_decoder/conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,col_decoder/conv2d_transpose/strided_slice_3
<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpEcol_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype02>
<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp
-col_decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput+col_decoder/conv2d_transpose/stack:output:0Dcol_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0Icol_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2/
-col_decoder/conv2d_transpose/conv2d_transposeã
3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp<col_decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp
$col_decoder/conv2d_transpose/BiasAddBiasAdd6col_decoder/conv2d_transpose/conv2d_transpose:output:0;col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2&
$col_decoder/conv2d_transpose/BiasAdd³
2col_decoder/conv2d_transpose/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ24
2col_decoder/conv2d_transpose/Max/reduction_indices
 col_decoder/conv2d_transpose/MaxMax-col_decoder/conv2d_transpose/BiasAdd:output:0;col_decoder/conv2d_transpose/Max/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2"
 col_decoder/conv2d_transpose/Maxá
 col_decoder/conv2d_transpose/subSub-col_decoder/conv2d_transpose/BiasAdd:output:0)col_decoder/conv2d_transpose/Max:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2"
 col_decoder/conv2d_transpose/sub­
 col_decoder/conv2d_transpose/ExpExp$col_decoder/conv2d_transpose/sub:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2"
 col_decoder/conv2d_transpose/Exp³
2col_decoder/conv2d_transpose/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ24
2col_decoder/conv2d_transpose/Sum/reduction_indicesû
 col_decoder/conv2d_transpose/SumSum$col_decoder/conv2d_transpose/Exp:y:0;col_decoder/conv2d_transpose/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2"
 col_decoder/conv2d_transpose/Sumä
$col_decoder/conv2d_transpose/truedivRealDiv$col_decoder/conv2d_transpose/Exp:y:0)col_decoder/conv2d_transpose/Sum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2&
$col_decoder/conv2d_transpose/truediv
IdentityIdentity(col_decoder/conv2d_transpose/truediv:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity

Identity_1Identity,table_decoder/conv2d_transpose_1/truediv:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*ê
_input_shapesØ
Õ:ÿÿÿÿÿÿÿÿÿ  :::::::::::::::::::::::::::::::::::::::::::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: 
ç

+__inference_block3_conv2_layer_call_fn_1390

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_13802
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ñ
ä
__inference__wrapped_model_1234
input_layer5
1model_block1_conv1_conv2d_readvariableop_resource6
2model_block1_conv1_biasadd_readvariableop_resource5
1model_block1_conv2_conv2d_readvariableop_resource6
2model_block1_conv2_biasadd_readvariableop_resource5
1model_block2_conv1_conv2d_readvariableop_resource6
2model_block2_conv1_biasadd_readvariableop_resource5
1model_block2_conv2_conv2d_readvariableop_resource6
2model_block2_conv2_biasadd_readvariableop_resource5
1model_block3_conv1_conv2d_readvariableop_resource6
2model_block3_conv1_biasadd_readvariableop_resource5
1model_block3_conv2_conv2d_readvariableop_resource6
2model_block3_conv2_biasadd_readvariableop_resource5
1model_block3_conv3_conv2d_readvariableop_resource6
2model_block3_conv3_biasadd_readvariableop_resource5
1model_block3_conv4_conv2d_readvariableop_resource6
2model_block3_conv4_biasadd_readvariableop_resource5
1model_block4_conv1_conv2d_readvariableop_resource6
2model_block4_conv1_biasadd_readvariableop_resource5
1model_block4_conv2_conv2d_readvariableop_resource6
2model_block4_conv2_biasadd_readvariableop_resource5
1model_block4_conv3_conv2d_readvariableop_resource6
2model_block4_conv3_biasadd_readvariableop_resource5
1model_block4_conv4_conv2d_readvariableop_resource6
2model_block4_conv4_biasadd_readvariableop_resource5
1model_block5_conv1_conv2d_readvariableop_resource6
2model_block5_conv1_biasadd_readvariableop_resource5
1model_block5_conv2_conv2d_readvariableop_resource6
2model_block5_conv2_biasadd_readvariableop_resource5
1model_block5_conv3_conv2d_readvariableop_resource6
2model_block5_conv3_biasadd_readvariableop_resource5
1model_block5_conv4_conv2d_readvariableop_resource6
2model_block5_conv4_biasadd_readvariableop_resource7
3model_block_6_conv_1_conv2d_readvariableop_resource8
4model_block_6_conv_1_biasadd_readvariableop_resource7
3model_block_6_conv_2_conv2d_readvariableop_resource8
4model_block_6_conv_2_biasadd_readvariableop_resource?
;model_table_decoder_conv2d_1_conv2d_readvariableop_resource@
<model_table_decoder_conv2d_1_biasadd_readvariableop_resource?
;model_table_decoder_conv2d_2_conv2d_readvariableop_resource@
<model_table_decoder_conv2d_2_biasadd_readvariableop_resourceS
Omodel_table_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceJ
Fmodel_table_decoder_conv2d_transpose_1_biasadd_readvariableop_resource;
7model_col_decoder_conv2d_conv2d_readvariableop_resource<
8model_col_decoder_conv2d_biasadd_readvariableop_resourceO
Kmodel_col_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resourceF
Bmodel_col_decoder_conv2d_transpose_biasadd_readvariableop_resource
identity

identity_1Î
(model/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(model/block1_conv1/Conv2D/ReadVariableOpã
model/block1_conv1/Conv2DConv2Dinput_layer0model/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingSAME*
strides
2
model/block1_conv1/Conv2DÅ
)model/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model/block1_conv1/BiasAdd/ReadVariableOpÖ
model/block1_conv1/BiasAddBiasAdd"model/block1_conv1/Conv2D:output:01model/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
model/block1_conv1/BiasAdd
model/block1_conv1/ReluRelu#model/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
model/block1_conv1/ReluÎ
(model/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(model/block1_conv2/Conv2D/ReadVariableOpý
model/block1_conv2/Conv2DConv2D%model/block1_conv1/Relu:activations:00model/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingSAME*
strides
2
model/block1_conv2/Conv2DÅ
)model/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model/block1_conv2/BiasAdd/ReadVariableOpÖ
model/block1_conv2/BiasAddBiasAdd"model/block1_conv2/Conv2D:output:01model/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
model/block1_conv2/BiasAdd
model/block1_conv2/ReluRelu#model/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
model/block1_conv2/Relu×
model/block1_pool/MaxPoolMaxPool%model/block1_conv2/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
model/block1_pool/MaxPoolÏ
(model/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(model/block2_conv1/Conv2D/ReadVariableOpû
model/block2_conv1/Conv2DConv2D"model/block1_pool/MaxPool:output:00model/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
model/block2_conv1/Conv2DÆ
)model/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block2_conv1/BiasAdd/ReadVariableOp×
model/block2_conv1/BiasAddBiasAdd"model/block2_conv1/Conv2D:output:01model/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2
model/block2_conv1/BiasAdd
model/block2_conv1/ReluRelu#model/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2
model/block2_conv1/ReluÐ
(model/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block2_conv2/Conv2D/ReadVariableOpþ
model/block2_conv2/Conv2DConv2D%model/block2_conv1/Relu:activations:00model/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
model/block2_conv2/Conv2DÆ
)model/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block2_conv2/BiasAdd/ReadVariableOp×
model/block2_conv2/BiasAddBiasAdd"model/block2_conv2/Conv2D:output:01model/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2
model/block2_conv2/BiasAdd
model/block2_conv2/ReluRelu#model/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2
model/block2_conv2/ReluØ
model/block2_pool/MaxPoolMaxPool%model/block2_conv2/Relu:activations:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
ksize
*
paddingVALID*
strides
2
model/block2_pool/MaxPoolÐ
(model/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block3_conv1/Conv2D/ReadVariableOpû
model/block3_conv1/Conv2DConv2D"model/block2_pool/MaxPool:output:00model/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
paddingSAME*
strides
2
model/block3_conv1/Conv2DÆ
)model/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block3_conv1/BiasAdd/ReadVariableOp×
model/block3_conv1/BiasAddBiasAdd"model/block3_conv1/Conv2D:output:01model/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
model/block3_conv1/BiasAdd
model/block3_conv1/ReluRelu#model/block3_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
model/block3_conv1/ReluÐ
(model/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block3_conv2/Conv2D/ReadVariableOpþ
model/block3_conv2/Conv2DConv2D%model/block3_conv1/Relu:activations:00model/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
paddingSAME*
strides
2
model/block3_conv2/Conv2DÆ
)model/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block3_conv2/BiasAdd/ReadVariableOp×
model/block3_conv2/BiasAddBiasAdd"model/block3_conv2/Conv2D:output:01model/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
model/block3_conv2/BiasAdd
model/block3_conv2/ReluRelu#model/block3_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
model/block3_conv2/ReluÐ
(model/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block3_conv3/Conv2D/ReadVariableOpþ
model/block3_conv3/Conv2DConv2D%model/block3_conv2/Relu:activations:00model/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
paddingSAME*
strides
2
model/block3_conv3/Conv2DÆ
)model/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block3_conv3/BiasAdd/ReadVariableOp×
model/block3_conv3/BiasAddBiasAdd"model/block3_conv3/Conv2D:output:01model/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
model/block3_conv3/BiasAdd
model/block3_conv3/ReluRelu#model/block3_conv3/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
model/block3_conv3/ReluÐ
(model/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block3_conv4/Conv2D/ReadVariableOpþ
model/block3_conv4/Conv2DConv2D%model/block3_conv3/Relu:activations:00model/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
paddingSAME*
strides
2
model/block3_conv4/Conv2DÆ
)model/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block3_conv4/BiasAdd/ReadVariableOp×
model/block3_conv4/BiasAddBiasAdd"model/block3_conv4/Conv2D:output:01model/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
model/block3_conv4/BiasAdd
model/block3_conv4/ReluRelu#model/block3_conv4/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
model/block3_conv4/ReluÖ
model/block3_pool/MaxPoolMaxPool%model/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
ksize
*
paddingVALID*
strides
2
model/block3_pool/MaxPoolÐ
(model/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block4_conv1/Conv2D/ReadVariableOpù
model/block4_conv1/Conv2DConv2D"model/block3_pool/MaxPool:output:00model/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
2
model/block4_conv1/Conv2DÆ
)model/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block4_conv1/BiasAdd/ReadVariableOpÕ
model/block4_conv1/BiasAddBiasAdd"model/block4_conv1/Conv2D:output:01model/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
model/block4_conv1/BiasAdd
model/block4_conv1/ReluRelu#model/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
model/block4_conv1/ReluÐ
(model/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block4_conv2/Conv2D/ReadVariableOpü
model/block4_conv2/Conv2DConv2D%model/block4_conv1/Relu:activations:00model/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
2
model/block4_conv2/Conv2DÆ
)model/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block4_conv2/BiasAdd/ReadVariableOpÕ
model/block4_conv2/BiasAddBiasAdd"model/block4_conv2/Conv2D:output:01model/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
model/block4_conv2/BiasAdd
model/block4_conv2/ReluRelu#model/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
model/block4_conv2/ReluÐ
(model/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block4_conv3/Conv2D/ReadVariableOpü
model/block4_conv3/Conv2DConv2D%model/block4_conv2/Relu:activations:00model/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
2
model/block4_conv3/Conv2DÆ
)model/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block4_conv3/BiasAdd/ReadVariableOpÕ
model/block4_conv3/BiasAddBiasAdd"model/block4_conv3/Conv2D:output:01model/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
model/block4_conv3/BiasAdd
model/block4_conv3/ReluRelu#model/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
model/block4_conv3/ReluÐ
(model/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block4_conv4/Conv2D/ReadVariableOpü
model/block4_conv4/Conv2DConv2D%model/block4_conv3/Relu:activations:00model/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
2
model/block4_conv4/Conv2DÆ
)model/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block4_conv4/BiasAdd/ReadVariableOpÕ
model/block4_conv4/BiasAddBiasAdd"model/block4_conv4/Conv2D:output:01model/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
model/block4_conv4/BiasAdd
model/block4_conv4/ReluRelu#model/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
model/block4_conv4/ReluÖ
model/block4_pool/MaxPoolMaxPool%model/block4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
2
model/block4_pool/MaxPoolÐ
(model/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block5_conv1/Conv2D/ReadVariableOpù
model/block5_conv1/Conv2DConv2D"model/block4_pool/MaxPool:output:00model/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
model/block5_conv1/Conv2DÆ
)model/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block5_conv1/BiasAdd/ReadVariableOpÕ
model/block5_conv1/BiasAddBiasAdd"model/block5_conv1/Conv2D:output:01model/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
model/block5_conv1/BiasAdd
model/block5_conv1/ReluRelu#model/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
model/block5_conv1/ReluÐ
(model/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block5_conv2/Conv2D/ReadVariableOpü
model/block5_conv2/Conv2DConv2D%model/block5_conv1/Relu:activations:00model/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
model/block5_conv2/Conv2DÆ
)model/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block5_conv2/BiasAdd/ReadVariableOpÕ
model/block5_conv2/BiasAddBiasAdd"model/block5_conv2/Conv2D:output:01model/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
model/block5_conv2/BiasAdd
model/block5_conv2/ReluRelu#model/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
model/block5_conv2/ReluÐ
(model/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block5_conv3/Conv2D/ReadVariableOpü
model/block5_conv3/Conv2DConv2D%model/block5_conv2/Relu:activations:00model/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
model/block5_conv3/Conv2DÆ
)model/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block5_conv3/BiasAdd/ReadVariableOpÕ
model/block5_conv3/BiasAddBiasAdd"model/block5_conv3/Conv2D:output:01model/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
model/block5_conv3/BiasAdd
model/block5_conv3/ReluRelu#model/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
model/block5_conv3/ReluÐ
(model/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(model/block5_conv4/Conv2D/ReadVariableOpü
model/block5_conv4/Conv2DConv2D%model/block5_conv3/Relu:activations:00model/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
model/block5_conv4/Conv2DÆ
)model/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model/block5_conv4/BiasAdd/ReadVariableOpÕ
model/block5_conv4/BiasAddBiasAdd"model/block5_conv4/Conv2D:output:01model/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
model/block5_conv4/BiasAdd
model/block5_conv4/ReluRelu#model/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
model/block5_conv4/ReluÖ
model/block5_pool/MaxPoolMaxPool%model/block5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
model/block5_pool/MaxPoolÖ
*model/block_6_conv_1/Conv2D/ReadVariableOpReadVariableOp3model_block_6_conv_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02,
*model/block_6_conv_1/Conv2D/ReadVariableOp
model/block_6_conv_1/Conv2DConv2D"model/block5_pool/MaxPool:output:02model/block_6_conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
model/block_6_conv_1/Conv2DÌ
+model/block_6_conv_1/BiasAdd/ReadVariableOpReadVariableOp4model_block_6_conv_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+model/block_6_conv_1/BiasAdd/ReadVariableOpÝ
model/block_6_conv_1/BiasAddBiasAdd$model/block_6_conv_1/Conv2D:output:03model/block_6_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/block_6_conv_1/BiasAdd 
model/block_6_conv_1/ReluRelu%model/block_6_conv_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/block_6_conv_1/Relu 
model/dropout/IdentityIdentity'model/block_6_conv_1/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dropout/IdentityÖ
*model/block_6_conv_2/Conv2D/ReadVariableOpReadVariableOp3model_block_6_conv_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02,
*model/block_6_conv_2/Conv2D/ReadVariableOpý
model/block_6_conv_2/Conv2DConv2Dmodel/dropout/Identity:output:02model/block_6_conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
model/block_6_conv_2/Conv2DÌ
+model/block_6_conv_2/BiasAdd/ReadVariableOpReadVariableOp4model_block_6_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+model/block_6_conv_2/BiasAdd/ReadVariableOpÝ
model/block_6_conv_2/BiasAddBiasAdd$model/block_6_conv_2/Conv2D:output:03model/block_6_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/block_6_conv_2/BiasAdd 
model/block_6_conv_2/ReluRelu%model/block_6_conv_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/block_6_conv_2/Relu¤
model/dropout_1/IdentityIdentity'model/block_6_conv_2/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dropout_1/Identityî
2model/table_decoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;model_table_decoder_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype024
2model/table_decoder/conv2d_1/Conv2D/ReadVariableOp
#model/table_decoder/conv2d_1/Conv2DConv2D!model/dropout_1/Identity:output:0:model/table_decoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#model/table_decoder/conv2d_1/Conv2Dä
3model/table_decoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp<model_table_decoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model/table_decoder/conv2d_1/BiasAdd/ReadVariableOpý
$model/table_decoder/conv2d_1/BiasAddBiasAdd,model/table_decoder/conv2d_1/Conv2D:output:0;model/table_decoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model/table_decoder/conv2d_1/BiasAdd¸
!model/table_decoder/conv2d_1/ReluRelu-model/table_decoder/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!model/table_decoder/conv2d_1/ReluÈ
&model/table_decoder/dropout_2/IdentityIdentity/model/table_decoder/conv2d_1/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model/table_decoder/dropout_2/Identityî
2model/table_decoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp;model_table_decoder_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype024
2model/table_decoder/conv2d_2/Conv2D/ReadVariableOp¥
#model/table_decoder/conv2d_2/Conv2DConv2D/model/table_decoder/dropout_2/Identity:output:0:model/table_decoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#model/table_decoder/conv2d_2/Conv2Dä
3model/table_decoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp<model_table_decoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model/table_decoder/conv2d_2/BiasAdd/ReadVariableOpý
$model/table_decoder/conv2d_2/BiasAddBiasAdd,model/table_decoder/conv2d_2/Conv2D:output:0;model/table_decoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model/table_decoder/conv2d_2/BiasAdd¸
!model/table_decoder/conv2d_2/ReluRelu-model/table_decoder/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!model/table_decoder/conv2d_2/Reluµ
)model/table_decoder/up_sampling2d_4/ShapeShape/model/table_decoder/conv2d_1/Relu:activations:0*
T0*
_output_shapes
:2+
)model/table_decoder/up_sampling2d_4/Shape¼
7model/table_decoder/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7model/table_decoder/up_sampling2d_4/strided_slice/stackÀ
9model/table_decoder/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model/table_decoder/up_sampling2d_4/strided_slice/stack_1À
9model/table_decoder/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model/table_decoder/up_sampling2d_4/strided_slice/stack_2¦
1model/table_decoder/up_sampling2d_4/strided_sliceStridedSlice2model/table_decoder/up_sampling2d_4/Shape:output:0@model/table_decoder/up_sampling2d_4/strided_slice/stack:output:0Bmodel/table_decoder/up_sampling2d_4/strided_slice/stack_1:output:0Bmodel/table_decoder/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:23
1model/table_decoder/up_sampling2d_4/strided_slice§
)model/table_decoder/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2+
)model/table_decoder/up_sampling2d_4/Constî
'model/table_decoder/up_sampling2d_4/mulMul:model/table_decoder/up_sampling2d_4/strided_slice:output:02model/table_decoder/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2)
'model/table_decoder/up_sampling2d_4/mul»
9model/table_decoder/up_sampling2d_4/resize/ResizeBilinearResizeBilinear/model/table_decoder/conv2d_1/Relu:activations:0+model/table_decoder/up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
half_pixel_centers(2;
9model/table_decoder/up_sampling2d_4/resize/ResizeBilinear
+model/table_decoder/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+model/table_decoder/concatenate/concat/axisÆ
&model/table_decoder/concatenate/concatConcatV2"model/block4_pool/MaxPool:output:0Jmodel/table_decoder/up_sampling2d_4/resize/ResizeBilinear:resized_images:04model/table_decoder/concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222(
&model/table_decoder/concatenate/concatµ
)model/table_decoder/up_sampling2d_5/ShapeShape/model/table_decoder/concatenate/concat:output:0*
T0*
_output_shapes
:2+
)model/table_decoder/up_sampling2d_5/Shape¼
7model/table_decoder/up_sampling2d_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7model/table_decoder/up_sampling2d_5/strided_slice/stackÀ
9model/table_decoder/up_sampling2d_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model/table_decoder/up_sampling2d_5/strided_slice/stack_1À
9model/table_decoder/up_sampling2d_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model/table_decoder/up_sampling2d_5/strided_slice/stack_2¦
1model/table_decoder/up_sampling2d_5/strided_sliceStridedSlice2model/table_decoder/up_sampling2d_5/Shape:output:0@model/table_decoder/up_sampling2d_5/strided_slice/stack:output:0Bmodel/table_decoder/up_sampling2d_5/strided_slice/stack_1:output:0Bmodel/table_decoder/up_sampling2d_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:23
1model/table_decoder/up_sampling2d_5/strided_slice§
)model/table_decoder/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2+
)model/table_decoder/up_sampling2d_5/Constî
'model/table_decoder/up_sampling2d_5/mulMul:model/table_decoder/up_sampling2d_5/strided_slice:output:02model/table_decoder/up_sampling2d_5/Const:output:0*
T0*
_output_shapes
:2)
'model/table_decoder/up_sampling2d_5/mul»
9model/table_decoder/up_sampling2d_5/resize/ResizeBilinearResizeBilinear/model/table_decoder/concatenate/concat:output:0+model/table_decoder/up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
half_pixel_centers(2;
9model/table_decoder/up_sampling2d_5/resize/ResizeBilinear 
-model/table_decoder/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-model/table_decoder/concatenate_1/concat/axisÌ
(model/table_decoder/concatenate_1/concatConcatV2"model/block3_pool/MaxPool:output:0Jmodel/table_decoder/up_sampling2d_5/resize/ResizeBilinear:resized_images:06model/table_decoder/concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2*
(model/table_decoder/concatenate_1/concat·
)model/table_decoder/up_sampling2d_6/ShapeShape1model/table_decoder/concatenate_1/concat:output:0*
T0*
_output_shapes
:2+
)model/table_decoder/up_sampling2d_6/Shape¼
7model/table_decoder/up_sampling2d_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7model/table_decoder/up_sampling2d_6/strided_slice/stackÀ
9model/table_decoder/up_sampling2d_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model/table_decoder/up_sampling2d_6/strided_slice/stack_1À
9model/table_decoder/up_sampling2d_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model/table_decoder/up_sampling2d_6/strided_slice/stack_2¦
1model/table_decoder/up_sampling2d_6/strided_sliceStridedSlice2model/table_decoder/up_sampling2d_6/Shape:output:0@model/table_decoder/up_sampling2d_6/strided_slice/stack:output:0Bmodel/table_decoder/up_sampling2d_6/strided_slice/stack_1:output:0Bmodel/table_decoder/up_sampling2d_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:23
1model/table_decoder/up_sampling2d_6/strided_slice§
)model/table_decoder/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2+
)model/table_decoder/up_sampling2d_6/Constî
'model/table_decoder/up_sampling2d_6/mulMul:model/table_decoder/up_sampling2d_6/strided_slice:output:02model/table_decoder/up_sampling2d_6/Const:output:0*
T0*
_output_shapes
:2)
'model/table_decoder/up_sampling2d_6/mulÔ
@model/table_decoder/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor1model/table_decoder/concatenate_1/concat:output:0+model/table_decoder/up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
half_pixel_centers(2B
@model/table_decoder/up_sampling2d_6/resize/ResizeNearestNeighbor×
)model/table_decoder/up_sampling2d_7/ShapeShapeQmodel/table_decoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2+
)model/table_decoder/up_sampling2d_7/Shape¼
7model/table_decoder/up_sampling2d_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7model/table_decoder/up_sampling2d_7/strided_slice/stackÀ
9model/table_decoder/up_sampling2d_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model/table_decoder/up_sampling2d_7/strided_slice/stack_1À
9model/table_decoder/up_sampling2d_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model/table_decoder/up_sampling2d_7/strided_slice/stack_2¦
1model/table_decoder/up_sampling2d_7/strided_sliceStridedSlice2model/table_decoder/up_sampling2d_7/Shape:output:0@model/table_decoder/up_sampling2d_7/strided_slice/stack:output:0Bmodel/table_decoder/up_sampling2d_7/strided_slice/stack_1:output:0Bmodel/table_decoder/up_sampling2d_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:23
1model/table_decoder/up_sampling2d_7/strided_slice§
)model/table_decoder/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2+
)model/table_decoder/up_sampling2d_7/Constî
'model/table_decoder/up_sampling2d_7/mulMul:model/table_decoder/up_sampling2d_7/strided_slice:output:02model/table_decoder/up_sampling2d_7/Const:output:0*
T0*
_output_shapes
:2)
'model/table_decoder/up_sampling2d_7/mulô
@model/table_decoder/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborQmodel/table_decoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0+model/table_decoder/up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2B
@model/table_decoder/up_sampling2d_7/resize/ResizeNearestNeighborÝ
,model/table_decoder/conv2d_transpose_1/ShapeShapeQmodel/table_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2.
,model/table_decoder/conv2d_transpose_1/ShapeÂ
:model/table_decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:model/table_decoder/conv2d_transpose_1/strided_slice/stackÆ
<model/table_decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<model/table_decoder/conv2d_transpose_1/strided_slice/stack_1Æ
<model/table_decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model/table_decoder/conv2d_transpose_1/strided_slice/stack_2Ì
4model/table_decoder/conv2d_transpose_1/strided_sliceStridedSlice5model/table_decoder/conv2d_transpose_1/Shape:output:0Cmodel/table_decoder/conv2d_transpose_1/strided_slice/stack:output:0Emodel/table_decoder/conv2d_transpose_1/strided_slice/stack_1:output:0Emodel/table_decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model/table_decoder/conv2d_transpose_1/strided_sliceÆ
<model/table_decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<model/table_decoder/conv2d_transpose_1/strided_slice_1/stackÊ
>model/table_decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>model/table_decoder/conv2d_transpose_1/strided_slice_1/stack_1Ê
>model/table_decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>model/table_decoder/conv2d_transpose_1/strided_slice_1/stack_2Ö
6model/table_decoder/conv2d_transpose_1/strided_slice_1StridedSlice5model/table_decoder/conv2d_transpose_1/Shape:output:0Emodel/table_decoder/conv2d_transpose_1/strided_slice_1/stack:output:0Gmodel/table_decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0Gmodel/table_decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6model/table_decoder/conv2d_transpose_1/strided_slice_1Æ
<model/table_decoder/conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<model/table_decoder/conv2d_transpose_1/strided_slice_2/stackÊ
>model/table_decoder/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>model/table_decoder/conv2d_transpose_1/strided_slice_2/stack_1Ê
>model/table_decoder/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>model/table_decoder/conv2d_transpose_1/strided_slice_2/stack_2Ö
6model/table_decoder/conv2d_transpose_1/strided_slice_2StridedSlice5model/table_decoder/conv2d_transpose_1/Shape:output:0Emodel/table_decoder/conv2d_transpose_1/strided_slice_2/stack:output:0Gmodel/table_decoder/conv2d_transpose_1/strided_slice_2/stack_1:output:0Gmodel/table_decoder/conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6model/table_decoder/conv2d_transpose_1/strided_slice_2
,model/table_decoder/conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,model/table_decoder/conv2d_transpose_1/mul/yø
*model/table_decoder/conv2d_transpose_1/mulMul?model/table_decoder/conv2d_transpose_1/strided_slice_1:output:05model/table_decoder/conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2,
*model/table_decoder/conv2d_transpose_1/mul¢
.model/table_decoder/conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.model/table_decoder/conv2d_transpose_1/mul_1/yþ
,model/table_decoder/conv2d_transpose_1/mul_1Mul?model/table_decoder/conv2d_transpose_1/strided_slice_2:output:07model/table_decoder/conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2.
,model/table_decoder/conv2d_transpose_1/mul_1¢
.model/table_decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :20
.model/table_decoder/conv2d_transpose_1/stack/3ì
,model/table_decoder/conv2d_transpose_1/stackPack=model/table_decoder/conv2d_transpose_1/strided_slice:output:0.model/table_decoder/conv2d_transpose_1/mul:z:00model/table_decoder/conv2d_transpose_1/mul_1:z:07model/table_decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2.
,model/table_decoder/conv2d_transpose_1/stackÆ
<model/table_decoder/conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<model/table_decoder/conv2d_transpose_1/strided_slice_3/stackÊ
>model/table_decoder/conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>model/table_decoder/conv2d_transpose_1/strided_slice_3/stack_1Ê
>model/table_decoder/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>model/table_decoder/conv2d_transpose_1/strided_slice_3/stack_2Ö
6model/table_decoder/conv2d_transpose_1/strided_slice_3StridedSlice5model/table_decoder/conv2d_transpose_1/stack:output:0Emodel/table_decoder/conv2d_transpose_1/strided_slice_3/stack:output:0Gmodel/table_decoder/conv2d_transpose_1/strided_slice_3/stack_1:output:0Gmodel/table_decoder/conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6model/table_decoder/conv2d_transpose_1/strided_slice_3©
Fmodel/table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpOmodel_table_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype02H
Fmodel/table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpÇ
7model/table_decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput5model/table_decoder/conv2d_transpose_1/stack:output:0Nmodel/table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Qmodel/table_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
29
7model/table_decoder/conv2d_transpose_1/conv2d_transpose
=model/table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpFmodel_table_decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=model/table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp°
.model/table_decoder/conv2d_transpose_1/BiasAddBiasAdd@model/table_decoder/conv2d_transpose_1/conv2d_transpose:output:0Emodel/table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  20
.model/table_decoder/conv2d_transpose_1/BiasAddÇ
<model/table_decoder/conv2d_transpose_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2>
<model/table_decoder/conv2d_transpose_1/Max/reduction_indices¬
*model/table_decoder/conv2d_transpose_1/MaxMax7model/table_decoder/conv2d_transpose_1/BiasAdd:output:0Emodel/table_decoder/conv2d_transpose_1/Max/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2,
*model/table_decoder/conv2d_transpose_1/Max
*model/table_decoder/conv2d_transpose_1/subSub7model/table_decoder/conv2d_transpose_1/BiasAdd:output:03model/table_decoder/conv2d_transpose_1/Max:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2,
*model/table_decoder/conv2d_transpose_1/subË
*model/table_decoder/conv2d_transpose_1/ExpExp.model/table_decoder/conv2d_transpose_1/sub:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2,
*model/table_decoder/conv2d_transpose_1/ExpÇ
<model/table_decoder/conv2d_transpose_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2>
<model/table_decoder/conv2d_transpose_1/Sum/reduction_indices£
*model/table_decoder/conv2d_transpose_1/SumSum.model/table_decoder/conv2d_transpose_1/Exp:y:0Emodel/table_decoder/conv2d_transpose_1/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2,
*model/table_decoder/conv2d_transpose_1/Sum
.model/table_decoder/conv2d_transpose_1/truedivRealDiv.model/table_decoder/conv2d_transpose_1/Exp:y:03model/table_decoder/conv2d_transpose_1/Sum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  20
.model/table_decoder/conv2d_transpose_1/truedivâ
.model/col_decoder/conv2d/Conv2D/ReadVariableOpReadVariableOp7model_col_decoder_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype020
.model/col_decoder/conv2d/Conv2D/ReadVariableOp
model/col_decoder/conv2d/Conv2DConv2D!model/dropout_1/Identity:output:06model/col_decoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2!
model/col_decoder/conv2d/Conv2DØ
/model/col_decoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp8model_col_decoder_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/model/col_decoder/conv2d/BiasAdd/ReadVariableOpí
 model/col_decoder/conv2d/BiasAddBiasAdd(model/col_decoder/conv2d/Conv2D:output:07model/col_decoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model/col_decoder/conv2d/BiasAdd¬
model/col_decoder/conv2d/ReluRelu)model/col_decoder/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/col_decoder/conv2d/Relu©
%model/col_decoder/up_sampling2d/ShapeShape+model/col_decoder/conv2d/Relu:activations:0*
T0*
_output_shapes
:2'
%model/col_decoder/up_sampling2d/Shape´
3model/col_decoder/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3model/col_decoder/up_sampling2d/strided_slice/stack¸
5model/col_decoder/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5model/col_decoder/up_sampling2d/strided_slice/stack_1¸
5model/col_decoder/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5model/col_decoder/up_sampling2d/strided_slice/stack_2
-model/col_decoder/up_sampling2d/strided_sliceStridedSlice.model/col_decoder/up_sampling2d/Shape:output:0<model/col_decoder/up_sampling2d/strided_slice/stack:output:0>model/col_decoder/up_sampling2d/strided_slice/stack_1:output:0>model/col_decoder/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2/
-model/col_decoder/up_sampling2d/strided_slice
%model/col_decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model/col_decoder/up_sampling2d/ConstÞ
#model/col_decoder/up_sampling2d/mulMul6model/col_decoder/up_sampling2d/strided_slice:output:0.model/col_decoder/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2%
#model/col_decoder/up_sampling2d/mul«
5model/col_decoder/up_sampling2d/resize/ResizeBilinearResizeBilinear+model/col_decoder/conv2d/Relu:activations:0'model/col_decoder/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
half_pixel_centers(27
5model/col_decoder/up_sampling2d/resize/ResizeBilinear
+model/col_decoder/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+model/col_decoder/concatenate_2/concat/axisÂ
&model/col_decoder/concatenate_2/concatConcatV2"model/block4_pool/MaxPool:output:0Fmodel/col_decoder/up_sampling2d/resize/ResizeBilinear:resized_images:04model/col_decoder/concatenate_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222(
&model/col_decoder/concatenate_2/concat±
'model/col_decoder/up_sampling2d_1/ShapeShape/model/col_decoder/concatenate_2/concat:output:0*
T0*
_output_shapes
:2)
'model/col_decoder/up_sampling2d_1/Shape¸
5model/col_decoder/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5model/col_decoder/up_sampling2d_1/strided_slice/stack¼
7model/col_decoder/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model/col_decoder/up_sampling2d_1/strided_slice/stack_1¼
7model/col_decoder/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model/col_decoder/up_sampling2d_1/strided_slice/stack_2
/model/col_decoder/up_sampling2d_1/strided_sliceStridedSlice0model/col_decoder/up_sampling2d_1/Shape:output:0>model/col_decoder/up_sampling2d_1/strided_slice/stack:output:0@model/col_decoder/up_sampling2d_1/strided_slice/stack_1:output:0@model/col_decoder/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:21
/model/col_decoder/up_sampling2d_1/strided_slice£
'model/col_decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2)
'model/col_decoder/up_sampling2d_1/Constæ
%model/col_decoder/up_sampling2d_1/mulMul8model/col_decoder/up_sampling2d_1/strided_slice:output:00model/col_decoder/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2'
%model/col_decoder/up_sampling2d_1/mulµ
7model/col_decoder/up_sampling2d_1/resize/ResizeBilinearResizeBilinear/model/col_decoder/concatenate_2/concat:output:0)model/col_decoder/up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
half_pixel_centers(29
7model/col_decoder/up_sampling2d_1/resize/ResizeBilinear
+model/col_decoder/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+model/col_decoder/concatenate_3/concat/axisÄ
&model/col_decoder/concatenate_3/concatConcatV2"model/block3_pool/MaxPool:output:0Hmodel/col_decoder/up_sampling2d_1/resize/ResizeBilinear:resized_images:04model/col_decoder/concatenate_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2(
&model/col_decoder/concatenate_3/concat±
'model/col_decoder/up_sampling2d_2/ShapeShape/model/col_decoder/concatenate_3/concat:output:0*
T0*
_output_shapes
:2)
'model/col_decoder/up_sampling2d_2/Shape¸
5model/col_decoder/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5model/col_decoder/up_sampling2d_2/strided_slice/stack¼
7model/col_decoder/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model/col_decoder/up_sampling2d_2/strided_slice/stack_1¼
7model/col_decoder/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model/col_decoder/up_sampling2d_2/strided_slice/stack_2
/model/col_decoder/up_sampling2d_2/strided_sliceStridedSlice0model/col_decoder/up_sampling2d_2/Shape:output:0>model/col_decoder/up_sampling2d_2/strided_slice/stack:output:0@model/col_decoder/up_sampling2d_2/strided_slice/stack_1:output:0@model/col_decoder/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:21
/model/col_decoder/up_sampling2d_2/strided_slice£
'model/col_decoder/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2)
'model/col_decoder/up_sampling2d_2/Constæ
%model/col_decoder/up_sampling2d_2/mulMul8model/col_decoder/up_sampling2d_2/strided_slice:output:00model/col_decoder/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2'
%model/col_decoder/up_sampling2d_2/mulÌ
>model/col_decoder/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor/model/col_decoder/concatenate_3/concat:output:0)model/col_decoder/up_sampling2d_2/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
half_pixel_centers(2@
>model/col_decoder/up_sampling2d_2/resize/ResizeNearestNeighborÑ
'model/col_decoder/up_sampling2d_3/ShapeShapeOmodel/col_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2)
'model/col_decoder/up_sampling2d_3/Shape¸
5model/col_decoder/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5model/col_decoder/up_sampling2d_3/strided_slice/stack¼
7model/col_decoder/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model/col_decoder/up_sampling2d_3/strided_slice/stack_1¼
7model/col_decoder/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model/col_decoder/up_sampling2d_3/strided_slice/stack_2
/model/col_decoder/up_sampling2d_3/strided_sliceStridedSlice0model/col_decoder/up_sampling2d_3/Shape:output:0>model/col_decoder/up_sampling2d_3/strided_slice/stack:output:0@model/col_decoder/up_sampling2d_3/strided_slice/stack_1:output:0@model/col_decoder/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:21
/model/col_decoder/up_sampling2d_3/strided_slice£
'model/col_decoder/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2)
'model/col_decoder/up_sampling2d_3/Constæ
%model/col_decoder/up_sampling2d_3/mulMul8model/col_decoder/up_sampling2d_3/strided_slice:output:00model/col_decoder/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2'
%model/col_decoder/up_sampling2d_3/mulì
>model/col_decoder/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborOmodel/col_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0)model/col_decoder/up_sampling2d_3/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2@
>model/col_decoder/up_sampling2d_3/resize/ResizeNearestNeighborÓ
(model/col_decoder/conv2d_transpose/ShapeShapeOmodel/col_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2*
(model/col_decoder/conv2d_transpose/Shapeº
6model/col_decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6model/col_decoder/conv2d_transpose/strided_slice/stack¾
8model/col_decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/col_decoder/conv2d_transpose/strided_slice/stack_1¾
8model/col_decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/col_decoder/conv2d_transpose/strided_slice/stack_2´
0model/col_decoder/conv2d_transpose/strided_sliceStridedSlice1model/col_decoder/conv2d_transpose/Shape:output:0?model/col_decoder/conv2d_transpose/strided_slice/stack:output:0Amodel/col_decoder/conv2d_transpose/strided_slice/stack_1:output:0Amodel/col_decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0model/col_decoder/conv2d_transpose/strided_slice¾
8model/col_decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8model/col_decoder/conv2d_transpose/strided_slice_1/stackÂ
:model/col_decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model/col_decoder/conv2d_transpose/strided_slice_1/stack_1Â
:model/col_decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model/col_decoder/conv2d_transpose/strided_slice_1/stack_2¾
2model/col_decoder/conv2d_transpose/strided_slice_1StridedSlice1model/col_decoder/conv2d_transpose/Shape:output:0Amodel/col_decoder/conv2d_transpose/strided_slice_1/stack:output:0Cmodel/col_decoder/conv2d_transpose/strided_slice_1/stack_1:output:0Cmodel/col_decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model/col_decoder/conv2d_transpose/strided_slice_1¾
8model/col_decoder/conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8model/col_decoder/conv2d_transpose/strided_slice_2/stackÂ
:model/col_decoder/conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model/col_decoder/conv2d_transpose/strided_slice_2/stack_1Â
:model/col_decoder/conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model/col_decoder/conv2d_transpose/strided_slice_2/stack_2¾
2model/col_decoder/conv2d_transpose/strided_slice_2StridedSlice1model/col_decoder/conv2d_transpose/Shape:output:0Amodel/col_decoder/conv2d_transpose/strided_slice_2/stack:output:0Cmodel/col_decoder/conv2d_transpose/strided_slice_2/stack_1:output:0Cmodel/col_decoder/conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model/col_decoder/conv2d_transpose/strided_slice_2
(model/col_decoder/conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(model/col_decoder/conv2d_transpose/mul/yè
&model/col_decoder/conv2d_transpose/mulMul;model/col_decoder/conv2d_transpose/strided_slice_1:output:01model/col_decoder/conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2(
&model/col_decoder/conv2d_transpose/mul
*model/col_decoder/conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*model/col_decoder/conv2d_transpose/mul_1/yî
(model/col_decoder/conv2d_transpose/mul_1Mul;model/col_decoder/conv2d_transpose/strided_slice_2:output:03model/col_decoder/conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2*
(model/col_decoder/conv2d_transpose/mul_1
*model/col_decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2,
*model/col_decoder/conv2d_transpose/stack/3Ô
(model/col_decoder/conv2d_transpose/stackPack9model/col_decoder/conv2d_transpose/strided_slice:output:0*model/col_decoder/conv2d_transpose/mul:z:0,model/col_decoder/conv2d_transpose/mul_1:z:03model/col_decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2*
(model/col_decoder/conv2d_transpose/stack¾
8model/col_decoder/conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model/col_decoder/conv2d_transpose/strided_slice_3/stackÂ
:model/col_decoder/conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model/col_decoder/conv2d_transpose/strided_slice_3/stack_1Â
:model/col_decoder/conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model/col_decoder/conv2d_transpose/strided_slice_3/stack_2¾
2model/col_decoder/conv2d_transpose/strided_slice_3StridedSlice1model/col_decoder/conv2d_transpose/stack:output:0Amodel/col_decoder/conv2d_transpose/strided_slice_3/stack:output:0Cmodel/col_decoder/conv2d_transpose/strided_slice_3/stack_1:output:0Cmodel/col_decoder/conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model/col_decoder/conv2d_transpose/strided_slice_3
Bmodel/col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpKmodel_col_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype02D
Bmodel/col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpµ
3model/col_decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput1model/col_decoder/conv2d_transpose/stack:output:0Jmodel/col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0Omodel/col_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
25
3model/col_decoder/conv2d_transpose/conv2d_transposeõ
9model/col_decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpBmodel_col_decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9model/col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp 
*model/col_decoder/conv2d_transpose/BiasAddBiasAdd<model/col_decoder/conv2d_transpose/conv2d_transpose:output:0Amodel/col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2,
*model/col_decoder/conv2d_transpose/BiasAdd¿
8model/col_decoder/conv2d_transpose/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2:
8model/col_decoder/conv2d_transpose/Max/reduction_indices
&model/col_decoder/conv2d_transpose/MaxMax3model/col_decoder/conv2d_transpose/BiasAdd:output:0Amodel/col_decoder/conv2d_transpose/Max/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2(
&model/col_decoder/conv2d_transpose/Maxù
&model/col_decoder/conv2d_transpose/subSub3model/col_decoder/conv2d_transpose/BiasAdd:output:0/model/col_decoder/conv2d_transpose/Max:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2(
&model/col_decoder/conv2d_transpose/sub¿
&model/col_decoder/conv2d_transpose/ExpExp*model/col_decoder/conv2d_transpose/sub:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2(
&model/col_decoder/conv2d_transpose/Exp¿
8model/col_decoder/conv2d_transpose/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2:
8model/col_decoder/conv2d_transpose/Sum/reduction_indices
&model/col_decoder/conv2d_transpose/SumSum*model/col_decoder/conv2d_transpose/Exp:y:0Amodel/col_decoder/conv2d_transpose/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2(
&model/col_decoder/conv2d_transpose/Sumü
*model/col_decoder/conv2d_transpose/truedivRealDiv*model/col_decoder/conv2d_transpose/Exp:y:0/model/col_decoder/conv2d_transpose/Sum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2,
*model/col_decoder/conv2d_transpose/truediv
IdentityIdentity.model/col_decoder/conv2d_transpose/truediv:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity

Identity_1Identity2model/table_decoder/conv2d_transpose_1/truediv:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*ê
_input_shapesØ
Õ:ÿÿÿÿÿÿÿÿÿ  :::::::::::::::::::::::::::::::::::::::::::::::^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%
_user_specified_nameInput_Layer:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: 
ÿ
F
*__inference_block5_pool_layer_call_fn_1646

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_16402
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À

°
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_1680

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp·
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
½

®
F__inference_block5_conv1_layer_call_and_return_conditional_losses_1558

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ç

+__inference_block4_conv4_layer_call_fn_1534

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_15242
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

J
.__inference_up_sampling2d_5_layer_call_fn_1921

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_19152
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½

®
F__inference_block3_conv4_layer_call_and_return_conditional_losses_1424

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ö
À
$__inference_model_layer_call_fn_4027

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity

identity_1¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  *P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_30022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*ê
_input_shapesØ
Õ:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: 

°
G__inference_table_decoder_layer_call_and_return_conditional_losses_4281
input_0
input_1
input_2+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource
identity²
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpÁ
conv2d_1/Conv2DConv2Dinput_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_1/Conv2D¨
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp­
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_1/BiasAdd|
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_1/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/dropout/Const¯
dropout_2/dropout/MulMulconv2d_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Mul}
dropout_2/dropout/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/ShapeÛ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2"
 dropout_2/dropout/GreaterEqual/yï
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dropout_2/dropout/GreaterEqual¦
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Cast«
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Mul_1²
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOpÕ
conv2d_2/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_2/Conv2D¨
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp­
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/Reluy
up_sampling2d_4/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2®
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mulë
%up_sampling2d_4/resize/ResizeBilinearResizeBilinearconv2d_1/Relu:activations:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
half_pixel_centers(2'
%up_sampling2d_4/resize/ResizeBilineart
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisÛ
concatenate/concatConcatV2input_16up_sampling2d_4/resize/ResizeBilinear:resized_images:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
concatenate/concaty
up_sampling2d_5/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/Shape
#up_sampling2d_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_5/strided_slice/stack
%up_sampling2d_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_5/strided_slice/stack_1
%up_sampling2d_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_5/strided_slice/stack_2®
up_sampling2d_5/strided_sliceStridedSliceup_sampling2d_5/Shape:output:0,up_sampling2d_5/strided_slice/stack:output:0.up_sampling2d_5/strided_slice/stack_1:output:0.up_sampling2d_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_5/strided_slice
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_5/Const
up_sampling2d_5/mulMul&up_sampling2d_5/strided_slice:output:0up_sampling2d_5/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/mulë
%up_sampling2d_5/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
half_pixel_centers(2'
%up_sampling2d_5/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisá
concatenate_1/concatConcatV2input_26up_sampling2d_5/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
concatenate_1/concat{
up_sampling2d_6/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/Shape
#up_sampling2d_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_6/strided_slice/stack
%up_sampling2d_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_6/strided_slice/stack_1
%up_sampling2d_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_6/strided_slice/stack_2®
up_sampling2d_6/strided_sliceStridedSliceup_sampling2d_6/Shape:output:0,up_sampling2d_6/strided_slice/stack:output:0.up_sampling2d_6/strided_slice/stack_1:output:0.up_sampling2d_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_6/strided_slice
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const
up_sampling2d_6/mulMul&up_sampling2d_6/strided_slice:output:0up_sampling2d_6/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/mul
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
half_pixel_centers(2.
,up_sampling2d_6/resize/ResizeNearestNeighbor
up_sampling2d_7/ShapeShape=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d_7/Shape
#up_sampling2d_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_7/strided_slice/stack
%up_sampling2d_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_7/strided_slice/stack_1
%up_sampling2d_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_7/strided_slice/stack_2®
up_sampling2d_7/strided_sliceStridedSliceup_sampling2d_7/Shape:output:0,up_sampling2d_7/strided_slice/stack:output:0.up_sampling2d_7/strided_slice/stack_1:output:0.up_sampling2d_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_7/strided_slice
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const
up_sampling2d_7/mulMul&up_sampling2d_7/strided_slice:output:0up_sampling2d_7/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_7/mul¤
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2.
,up_sampling2d_7/resize/ResizeNearestNeighbor¡
conv2d_transpose_1/ShapeShape=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2Ô
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stack¢
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1¢
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2Þ
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_2/stack¢
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1¢
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2Þ
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y¨
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulz
conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul_1/y®
conv2d_transpose_1/mul_1Mul+conv2d_transpose_1/strided_slice_2:output:0#conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mul_1z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3ô
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/mul:z:0conv2d_transpose_1/mul_1:z:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack
(conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_3/stack¢
*conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_1¢
*conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_2Þ
"conv2d_transpose_1/strided_slice_3StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_3/stack:output:03conv2d_transpose_1/strided_slice_3/stack_1:output:03conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_3í
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpã
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transposeÅ
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpà
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/BiasAdd
(conv2d_transpose_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(conv2d_transpose_1/Max/reduction_indicesÜ
conv2d_transpose_1/MaxMax#conv2d_transpose_1/BiasAdd:output:01conv2d_transpose_1/Max/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2
conv2d_transpose_1/Max¹
conv2d_transpose_1/subSub#conv2d_transpose_1/BiasAdd:output:0conv2d_transpose_1/Max:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/sub
conv2d_transpose_1/ExpExpconv2d_transpose_1/sub:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/Exp
(conv2d_transpose_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(conv2d_transpose_1/Sum/reduction_indicesÓ
conv2d_transpose_1/SumSumconv2d_transpose_1/Exp:y:01conv2d_transpose_1/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2
conv2d_transpose_1/Sum¼
conv2d_transpose_1/truedivRealDivconv2d_transpose_1/Exp:y:0conv2d_transpose_1/Sum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/truediv|
IdentityIdentityconv2d_transpose_1/truediv:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*
_input_shapesn
l:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22:ÿÿÿÿÿÿÿÿÿdd:::::::Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/0:YU
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input/1:YU
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!
_user_specified_name	input/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ç

+__inference_block4_conv1_layer_call_fn_1468

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_14582
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
½

®
F__inference_block5_conv4_layer_call_and_return_conditional_losses_1624

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

D
(__inference_dropout_1_layer_call_fn_4081

inputs
identity¨
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_21562
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_2156

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
e
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_1896

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÀ
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeBilinear
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
e
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1744

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÀ
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeBilinear
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_1540

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç

+__inference_block5_conv2_layer_call_fn_1590

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_15802
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ýÂ
©
 __inference__traced_restore_4728
file_prefix(
$assignvariableop_block1_conv1_kernel(
$assignvariableop_1_block1_conv1_bias*
&assignvariableop_2_block1_conv2_kernel(
$assignvariableop_3_block1_conv2_bias*
&assignvariableop_4_block2_conv1_kernel(
$assignvariableop_5_block2_conv1_bias*
&assignvariableop_6_block2_conv2_kernel(
$assignvariableop_7_block2_conv2_bias*
&assignvariableop_8_block3_conv1_kernel(
$assignvariableop_9_block3_conv1_bias+
'assignvariableop_10_block3_conv2_kernel)
%assignvariableop_11_block3_conv2_bias+
'assignvariableop_12_block3_conv3_kernel)
%assignvariableop_13_block3_conv3_bias+
'assignvariableop_14_block3_conv4_kernel)
%assignvariableop_15_block3_conv4_bias+
'assignvariableop_16_block4_conv1_kernel)
%assignvariableop_17_block4_conv1_bias+
'assignvariableop_18_block4_conv2_kernel)
%assignvariableop_19_block4_conv2_bias+
'assignvariableop_20_block4_conv3_kernel)
%assignvariableop_21_block4_conv3_bias+
'assignvariableop_22_block4_conv4_kernel)
%assignvariableop_23_block4_conv4_bias+
'assignvariableop_24_block5_conv1_kernel)
%assignvariableop_25_block5_conv1_bias+
'assignvariableop_26_block5_conv2_kernel)
%assignvariableop_27_block5_conv2_bias+
'assignvariableop_28_block5_conv3_kernel)
%assignvariableop_29_block5_conv3_bias+
'assignvariableop_30_block5_conv4_kernel)
%assignvariableop_31_block5_conv4_bias-
)assignvariableop_32_block_6_conv_1_kernel+
'assignvariableop_33_block_6_conv_1_bias-
)assignvariableop_34_block_6_conv_2_kernel+
'assignvariableop_35_block_6_conv_2_bias1
-assignvariableop_36_col_decoder_conv2d_kernel/
+assignvariableop_37_col_decoder_conv2d_bias;
7assignvariableop_38_col_decoder_conv2d_transpose_kernel9
5assignvariableop_39_col_decoder_conv2d_transpose_bias5
1assignvariableop_40_table_decoder_conv2d_1_kernel3
/assignvariableop_41_table_decoder_conv2d_1_bias5
1assignvariableop_42_table_decoder_conv2d_2_kernel3
/assignvariableop_43_table_decoder_conv2d_2_bias?
;assignvariableop_44_table_decoder_conv2d_transpose_1_kernel=
9assignvariableop_45_table_decoder_conv2d_transpose_1_bias
identity_47¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1ï
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*û
valueñBî.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesê
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Î
_output_shapes»
¸::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp$assignvariableop_block1_conv1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp$assignvariableop_1_block1_conv1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp&assignvariableop_2_block1_conv2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp$assignvariableop_3_block1_conv2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOp&assignvariableop_4_block2_conv1_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOp$assignvariableop_5_block2_conv1_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block2_conv2_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block2_conv2_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block3_conv1_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block3_conv1_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10 
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block3_conv2_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOp%assignvariableop_11_block3_conv2_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12 
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block3_conv3_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOp%assignvariableop_13_block3_conv3_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14 
AssignVariableOp_14AssignVariableOp'assignvariableop_14_block3_conv4_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOp%assignvariableop_15_block3_conv4_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16 
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block4_conv1_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17
AssignVariableOp_17AssignVariableOp%assignvariableop_17_block4_conv1_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18 
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block4_conv2_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19
AssignVariableOp_19AssignVariableOp%assignvariableop_19_block4_conv2_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20 
AssignVariableOp_20AssignVariableOp'assignvariableop_20_block4_conv3_kernelIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21
AssignVariableOp_21AssignVariableOp%assignvariableop_21_block4_conv3_biasIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22 
AssignVariableOp_22AssignVariableOp'assignvariableop_22_block4_conv4_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23
AssignVariableOp_23AssignVariableOp%assignvariableop_23_block4_conv4_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24 
AssignVariableOp_24AssignVariableOp'assignvariableop_24_block5_conv1_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25
AssignVariableOp_25AssignVariableOp%assignvariableop_25_block5_conv1_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26 
AssignVariableOp_26AssignVariableOp'assignvariableop_26_block5_conv2_kernelIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27
AssignVariableOp_27AssignVariableOp%assignvariableop_27_block5_conv2_biasIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28 
AssignVariableOp_28AssignVariableOp'assignvariableop_28_block5_conv3_kernelIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29
AssignVariableOp_29AssignVariableOp%assignvariableop_29_block5_conv3_biasIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30 
AssignVariableOp_30AssignVariableOp'assignvariableop_30_block5_conv4_kernelIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31
AssignVariableOp_31AssignVariableOp%assignvariableop_31_block5_conv4_biasIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32¢
AssignVariableOp_32AssignVariableOp)assignvariableop_32_block_6_conv_1_kernelIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33 
AssignVariableOp_33AssignVariableOp'assignvariableop_33_block_6_conv_1_biasIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34¢
AssignVariableOp_34AssignVariableOp)assignvariableop_34_block_6_conv_2_kernelIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35 
AssignVariableOp_35AssignVariableOp'assignvariableop_35_block_6_conv_2_biasIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36¦
AssignVariableOp_36AssignVariableOp-assignvariableop_36_col_decoder_conv2d_kernelIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37¤
AssignVariableOp_37AssignVariableOp+assignvariableop_37_col_decoder_conv2d_biasIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOp7assignvariableop_38_col_decoder_conv2d_transpose_kernelIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39®
AssignVariableOp_39AssignVariableOp5assignvariableop_39_col_decoder_conv2d_transpose_biasIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40ª
AssignVariableOp_40AssignVariableOp1assignvariableop_40_table_decoder_conv2d_1_kernelIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41¨
AssignVariableOp_41AssignVariableOp/assignvariableop_41_table_decoder_conv2d_1_biasIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42ª
AssignVariableOp_42AssignVariableOp1assignvariableop_42_table_decoder_conv2d_2_kernelIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43¨
AssignVariableOp_43AssignVariableOp/assignvariableop_43_table_decoder_conv2d_2_biasIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44´
AssignVariableOp_44AssignVariableOp;assignvariableop_44_table_decoder_conv2d_transpose_1_kernelIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45²
AssignVariableOp_45AssignVariableOp9assignvariableop_45_table_decoder_conv2d_transpose_1_biasIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÒ
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_46ß
Identity_47IdentityIdentity_46:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_47"#
identity_47Identity_47:output:0*Ï
_input_shapes½
º: ::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: 

J
.__inference_up_sampling2d_1_layer_call_fn_1750

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_17442
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
ë
?__inference_model_layer_call_and_return_conditional_losses_3525

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block3_conv3_conv2d_readvariableop_resource0
,block3_conv3_biasadd_readvariableop_resource/
+block3_conv4_conv2d_readvariableop_resource0
,block3_conv4_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block4_conv3_conv2d_readvariableop_resource0
,block4_conv3_biasadd_readvariableop_resource/
+block4_conv4_conv2d_readvariableop_resource0
,block4_conv4_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource/
+block5_conv3_conv2d_readvariableop_resource0
,block5_conv3_biasadd_readvariableop_resource/
+block5_conv4_conv2d_readvariableop_resource0
,block5_conv4_biasadd_readvariableop_resource1
-block_6_conv_1_conv2d_readvariableop_resource2
.block_6_conv_1_biasadd_readvariableop_resource1
-block_6_conv_2_conv2d_readvariableop_resource2
.block_6_conv_2_biasadd_readvariableop_resource9
5table_decoder_conv2d_1_conv2d_readvariableop_resource:
6table_decoder_conv2d_1_biasadd_readvariableop_resource9
5table_decoder_conv2d_2_conv2d_readvariableop_resource:
6table_decoder_conv2d_2_biasadd_readvariableop_resourceM
Itable_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceD
@table_decoder_conv2d_transpose_1_biasadd_readvariableop_resource5
1col_decoder_conv2d_conv2d_readvariableop_resource6
2col_decoder_conv2d_biasadd_readvariableop_resourceI
Ecol_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource@
<col_decoder_conv2d_transpose_biasadd_readvariableop_resource
identity

identity_1¼
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOpÌ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingSAME*
strides
2
block1_conv1/Conv2D³
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp¾
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
block1_conv1/BiasAdd
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
block1_conv1/Relu¼
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOpå
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingSAME*
strides
2
block1_conv2/Conv2D³
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp¾
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
block1_conv2/BiasAdd
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
block1_conv2/ReluÅ
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool½
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"block2_conv1/Conv2D/ReadVariableOpã
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block2_conv1/Conv2D´
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp¿
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2
block2_conv1/BiasAdd
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2
block2_conv1/Relu¾
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block2_conv2/Conv2D/ReadVariableOpæ
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block2_conv2/Conv2D´
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp¿
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2
block2_conv2/BiasAdd
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2
block2_conv2/ReluÆ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool¾
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv1/Conv2D/ReadVariableOpã
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
paddingSAME*
strides
2
block3_conv1/Conv2D´
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp¿
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv1/BiasAdd
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv1/Relu¾
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv2/Conv2D/ReadVariableOpæ
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
paddingSAME*
strides
2
block3_conv2/Conv2D´
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp¿
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv2/BiasAdd
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv2/Relu¾
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv3/Conv2D/ReadVariableOpæ
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
paddingSAME*
strides
2
block3_conv3/Conv2D´
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp¿
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv3/BiasAdd
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv3/Relu¾
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block3_conv4/Conv2D/ReadVariableOpæ
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
paddingSAME*
strides
2
block3_conv4/Conv2D´
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block3_conv4/BiasAdd/ReadVariableOp¿
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv4/BiasAdd
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ2
block3_conv4/ReluÄ
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool¾
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv1/Conv2D/ReadVariableOpá
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
2
block4_conv1/Conv2D´
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp½
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv1/BiasAdd
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv1/Relu¾
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv2/Conv2D/ReadVariableOpä
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
2
block4_conv2/Conv2D´
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp½
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv2/BiasAdd
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv2/Relu¾
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv3/Conv2D/ReadVariableOpä
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
2
block4_conv3/Conv2D´
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp½
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv3/BiasAdd
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv3/Relu¾
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block4_conv4/Conv2D/ReadVariableOpä
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
2
block4_conv4/Conv2D´
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block4_conv4/BiasAdd/ReadVariableOp½
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv4/BiasAdd
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
block4_conv4/ReluÄ
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool¾
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5_conv1/Conv2D/ReadVariableOpá
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
block5_conv1/Conv2D´
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp½
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv1/BiasAdd
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv1/Relu¾
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5_conv2/Conv2D/ReadVariableOpä
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
block5_conv2/Conv2D´
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOp½
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv2/BiasAdd
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv2/Relu¾
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5_conv3/Conv2D/ReadVariableOpä
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
block5_conv3/Conv2D´
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOp½
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv3/BiasAdd
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv3/Relu¾
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block5_conv4/Conv2D/ReadVariableOpä
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
block5_conv4/Conv2D´
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block5_conv4/BiasAdd/ReadVariableOp½
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv4/BiasAdd
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
block5_conv4/ReluÄ
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPoolÄ
$block_6_conv_1/Conv2D/ReadVariableOpReadVariableOp-block_6_conv_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02&
$block_6_conv_1/Conv2D/ReadVariableOpè
block_6_conv_1/Conv2DConv2Dblock5_pool/MaxPool:output:0,block_6_conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
block_6_conv_1/Conv2Dº
%block_6_conv_1/BiasAdd/ReadVariableOpReadVariableOp.block_6_conv_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%block_6_conv_1/BiasAdd/ReadVariableOpÅ
block_6_conv_1/BiasAddBiasAddblock_6_conv_1/Conv2D:output:0-block_6_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block_6_conv_1/BiasAdd
block_6_conv_1/ReluRelublock_6_conv_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block_6_conv_1/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/dropout/Const¯
dropout/dropout/MulMul!block_6_conv_1/Relu:activations:0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout/Mul
dropout/dropout/ShapeShape!block_6_conv_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÕ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2 
dropout/dropout/GreaterEqual/yç
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout/GreaterEqual 
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout/Cast£
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout/Mul_1Ä
$block_6_conv_2/Conv2D/ReadVariableOpReadVariableOp-block_6_conv_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02&
$block_6_conv_2/Conv2D/ReadVariableOpå
block_6_conv_2/Conv2DConv2Ddropout/dropout/Mul_1:z:0,block_6_conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
block_6_conv_2/Conv2Dº
%block_6_conv_2/BiasAdd/ReadVariableOpReadVariableOp.block_6_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%block_6_conv_2/BiasAdd/ReadVariableOpÅ
block_6_conv_2/BiasAddBiasAddblock_6_conv_2/Conv2D:output:0-block_6_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block_6_conv_2/BiasAdd
block_6_conv_2/ReluRelublock_6_conv_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block_6_conv_2/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/dropout/Constµ
dropout_1/dropout/MulMul!block_6_conv_2/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Mul
dropout_1/dropout/ShapeShape!block_6_conv_2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeÛ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2"
 dropout_1/dropout/GreaterEqual/yï
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dropout_1/dropout/GreaterEqual¦
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Cast«
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Mul_1Ü
,table_decoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5table_decoder_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,table_decoder/conv2d_1/Conv2D/ReadVariableOpÿ
table_decoder/conv2d_1/Conv2DConv2Ddropout_1/dropout/Mul_1:z:04table_decoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
table_decoder/conv2d_1/Conv2DÒ
-table_decoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6table_decoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-table_decoder/conv2d_1/BiasAdd/ReadVariableOpå
table_decoder/conv2d_1/BiasAddBiasAdd&table_decoder/conv2d_1/Conv2D:output:05table_decoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
table_decoder/conv2d_1/BiasAdd¦
table_decoder/conv2d_1/ReluRelu'table_decoder/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
table_decoder/conv2d_1/Relu
%table_decoder/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%table_decoder/dropout_2/dropout/Constç
#table_decoder/dropout_2/dropout/MulMul)table_decoder/conv2d_1/Relu:activations:0.table_decoder/dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#table_decoder/dropout_2/dropout/Mul§
%table_decoder/dropout_2/dropout/ShapeShape)table_decoder/conv2d_1/Relu:activations:0*
T0*
_output_shapes
:2'
%table_decoder/dropout_2/dropout/Shape
<table_decoder/dropout_2/dropout/random_uniform/RandomUniformRandomUniform.table_decoder/dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02>
<table_decoder/dropout_2/dropout/random_uniform/RandomUniform¥
.table_decoder/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>20
.table_decoder/dropout_2/dropout/GreaterEqual/y§
,table_decoder/dropout_2/dropout/GreaterEqualGreaterEqualEtable_decoder/dropout_2/dropout/random_uniform/RandomUniform:output:07table_decoder/dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,table_decoder/dropout_2/dropout/GreaterEqualÐ
$table_decoder/dropout_2/dropout/CastCast0table_decoder/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$table_decoder/dropout_2/dropout/Castã
%table_decoder/dropout_2/dropout/Mul_1Mul'table_decoder/dropout_2/dropout/Mul:z:0(table_decoder/dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%table_decoder/dropout_2/dropout/Mul_1Ü
,table_decoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5table_decoder_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,table_decoder/conv2d_2/Conv2D/ReadVariableOp
table_decoder/conv2d_2/Conv2DConv2D)table_decoder/dropout_2/dropout/Mul_1:z:04table_decoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
table_decoder/conv2d_2/Conv2DÒ
-table_decoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6table_decoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-table_decoder/conv2d_2/BiasAdd/ReadVariableOpå
table_decoder/conv2d_2/BiasAddBiasAdd&table_decoder/conv2d_2/Conv2D:output:05table_decoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
table_decoder/conv2d_2/BiasAdd¦
table_decoder/conv2d_2/ReluRelu'table_decoder/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
table_decoder/conv2d_2/Relu£
#table_decoder/up_sampling2d_4/ShapeShape)table_decoder/conv2d_1/Relu:activations:0*
T0*
_output_shapes
:2%
#table_decoder/up_sampling2d_4/Shape°
1table_decoder/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1table_decoder/up_sampling2d_4/strided_slice/stack´
3table_decoder/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_4/strided_slice/stack_1´
3table_decoder/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_4/strided_slice/stack_2
+table_decoder/up_sampling2d_4/strided_sliceStridedSlice,table_decoder/up_sampling2d_4/Shape:output:0:table_decoder/up_sampling2d_4/strided_slice/stack:output:0<table_decoder/up_sampling2d_4/strided_slice/stack_1:output:0<table_decoder/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2-
+table_decoder/up_sampling2d_4/strided_slice
#table_decoder/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#table_decoder/up_sampling2d_4/ConstÖ
!table_decoder/up_sampling2d_4/mulMul4table_decoder/up_sampling2d_4/strided_slice:output:0,table_decoder/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_4/mul£
3table_decoder/up_sampling2d_4/resize/ResizeBilinearResizeBilinear)table_decoder/conv2d_1/Relu:activations:0%table_decoder/up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
half_pixel_centers(25
3table_decoder/up_sampling2d_4/resize/ResizeBilinear
%table_decoder/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%table_decoder/concatenate/concat/axis¨
 table_decoder/concatenate/concatConcatV2block4_pool/MaxPool:output:0Dtable_decoder/up_sampling2d_4/resize/ResizeBilinear:resized_images:0.table_decoder/concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222"
 table_decoder/concatenate/concat£
#table_decoder/up_sampling2d_5/ShapeShape)table_decoder/concatenate/concat:output:0*
T0*
_output_shapes
:2%
#table_decoder/up_sampling2d_5/Shape°
1table_decoder/up_sampling2d_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1table_decoder/up_sampling2d_5/strided_slice/stack´
3table_decoder/up_sampling2d_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_5/strided_slice/stack_1´
3table_decoder/up_sampling2d_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_5/strided_slice/stack_2
+table_decoder/up_sampling2d_5/strided_sliceStridedSlice,table_decoder/up_sampling2d_5/Shape:output:0:table_decoder/up_sampling2d_5/strided_slice/stack:output:0<table_decoder/up_sampling2d_5/strided_slice/stack_1:output:0<table_decoder/up_sampling2d_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2-
+table_decoder/up_sampling2d_5/strided_slice
#table_decoder/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#table_decoder/up_sampling2d_5/ConstÖ
!table_decoder/up_sampling2d_5/mulMul4table_decoder/up_sampling2d_5/strided_slice:output:0,table_decoder/up_sampling2d_5/Const:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_5/mul£
3table_decoder/up_sampling2d_5/resize/ResizeBilinearResizeBilinear)table_decoder/concatenate/concat:output:0%table_decoder/up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
half_pixel_centers(25
3table_decoder/up_sampling2d_5/resize/ResizeBilinear
'table_decoder/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2)
'table_decoder/concatenate_1/concat/axis®
"table_decoder/concatenate_1/concatConcatV2block3_pool/MaxPool:output:0Dtable_decoder/up_sampling2d_5/resize/ResizeBilinear:resized_images:00table_decoder/concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2$
"table_decoder/concatenate_1/concat¥
#table_decoder/up_sampling2d_6/ShapeShape+table_decoder/concatenate_1/concat:output:0*
T0*
_output_shapes
:2%
#table_decoder/up_sampling2d_6/Shape°
1table_decoder/up_sampling2d_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1table_decoder/up_sampling2d_6/strided_slice/stack´
3table_decoder/up_sampling2d_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_6/strided_slice/stack_1´
3table_decoder/up_sampling2d_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_6/strided_slice/stack_2
+table_decoder/up_sampling2d_6/strided_sliceStridedSlice,table_decoder/up_sampling2d_6/Shape:output:0:table_decoder/up_sampling2d_6/strided_slice/stack:output:0<table_decoder/up_sampling2d_6/strided_slice/stack_1:output:0<table_decoder/up_sampling2d_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2-
+table_decoder/up_sampling2d_6/strided_slice
#table_decoder/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#table_decoder/up_sampling2d_6/ConstÖ
!table_decoder/up_sampling2d_6/mulMul4table_decoder/up_sampling2d_6/strided_slice:output:0,table_decoder/up_sampling2d_6/Const:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_6/mul¼
:table_decoder/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor+table_decoder/concatenate_1/concat:output:0%table_decoder/up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
half_pixel_centers(2<
:table_decoder/up_sampling2d_6/resize/ResizeNearestNeighborÅ
#table_decoder/up_sampling2d_7/ShapeShapeKtable_decoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2%
#table_decoder/up_sampling2d_7/Shape°
1table_decoder/up_sampling2d_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1table_decoder/up_sampling2d_7/strided_slice/stack´
3table_decoder/up_sampling2d_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_7/strided_slice/stack_1´
3table_decoder/up_sampling2d_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3table_decoder/up_sampling2d_7/strided_slice/stack_2
+table_decoder/up_sampling2d_7/strided_sliceStridedSlice,table_decoder/up_sampling2d_7/Shape:output:0:table_decoder/up_sampling2d_7/strided_slice/stack:output:0<table_decoder/up_sampling2d_7/strided_slice/stack_1:output:0<table_decoder/up_sampling2d_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2-
+table_decoder/up_sampling2d_7/strided_slice
#table_decoder/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#table_decoder/up_sampling2d_7/ConstÖ
!table_decoder/up_sampling2d_7/mulMul4table_decoder/up_sampling2d_7/strided_slice:output:0,table_decoder/up_sampling2d_7/Const:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_7/mulÜ
:table_decoder/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborKtable_decoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0%table_decoder/up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2<
:table_decoder/up_sampling2d_7/resize/ResizeNearestNeighborË
&table_decoder/conv2d_transpose_1/ShapeShapeKtable_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2(
&table_decoder/conv2d_transpose_1/Shape¶
4table_decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4table_decoder/conv2d_transpose_1/strided_slice/stackº
6table_decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6table_decoder/conv2d_transpose_1/strided_slice/stack_1º
6table_decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6table_decoder/conv2d_transpose_1/strided_slice/stack_2¨
.table_decoder/conv2d_transpose_1/strided_sliceStridedSlice/table_decoder/conv2d_transpose_1/Shape:output:0=table_decoder/conv2d_transpose_1/strided_slice/stack:output:0?table_decoder/conv2d_transpose_1/strided_slice/stack_1:output:0?table_decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.table_decoder/conv2d_transpose_1/strided_sliceº
6table_decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6table_decoder/conv2d_transpose_1/strided_slice_1/stack¾
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_1¾
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_2²
0table_decoder/conv2d_transpose_1/strided_slice_1StridedSlice/table_decoder/conv2d_transpose_1/Shape:output:0?table_decoder/conv2d_transpose_1/strided_slice_1/stack:output:0Atable_decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0Atable_decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0table_decoder/conv2d_transpose_1/strided_slice_1º
6table_decoder/conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6table_decoder/conv2d_transpose_1/strided_slice_2/stack¾
8table_decoder/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_2/stack_1¾
8table_decoder/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_2/stack_2²
0table_decoder/conv2d_transpose_1/strided_slice_2StridedSlice/table_decoder/conv2d_transpose_1/Shape:output:0?table_decoder/conv2d_transpose_1/strided_slice_2/stack:output:0Atable_decoder/conv2d_transpose_1/strided_slice_2/stack_1:output:0Atable_decoder/conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0table_decoder/conv2d_transpose_1/strided_slice_2
&table_decoder/conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&table_decoder/conv2d_transpose_1/mul/yà
$table_decoder/conv2d_transpose_1/mulMul9table_decoder/conv2d_transpose_1/strided_slice_1:output:0/table_decoder/conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2&
$table_decoder/conv2d_transpose_1/mul
(table_decoder/conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(table_decoder/conv2d_transpose_1/mul_1/yæ
&table_decoder/conv2d_transpose_1/mul_1Mul9table_decoder/conv2d_transpose_1/strided_slice_2:output:01table_decoder/conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2(
&table_decoder/conv2d_transpose_1/mul_1
(table_decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(table_decoder/conv2d_transpose_1/stack/3È
&table_decoder/conv2d_transpose_1/stackPack7table_decoder/conv2d_transpose_1/strided_slice:output:0(table_decoder/conv2d_transpose_1/mul:z:0*table_decoder/conv2d_transpose_1/mul_1:z:01table_decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&table_decoder/conv2d_transpose_1/stackº
6table_decoder/conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6table_decoder/conv2d_transpose_1/strided_slice_3/stack¾
8table_decoder/conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_3/stack_1¾
8table_decoder/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_3/stack_2²
0table_decoder/conv2d_transpose_1/strided_slice_3StridedSlice/table_decoder/conv2d_transpose_1/stack:output:0?table_decoder/conv2d_transpose_1/strided_slice_3/stack:output:0Atable_decoder/conv2d_transpose_1/strided_slice_3/stack_1:output:0Atable_decoder/conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0table_decoder/conv2d_transpose_1/strided_slice_3
@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpItable_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype02B
@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp©
1table_decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput/table_decoder/conv2d_transpose_1/stack:output:0Htable_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Ktable_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
23
1table_decoder/conv2d_transpose_1/conv2d_transposeï
7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp@table_decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp
(table_decoder/conv2d_transpose_1/BiasAddBiasAdd:table_decoder/conv2d_transpose_1/conv2d_transpose:output:0?table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2*
(table_decoder/conv2d_transpose_1/BiasAdd»
6table_decoder/conv2d_transpose_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ28
6table_decoder/conv2d_transpose_1/Max/reduction_indices
$table_decoder/conv2d_transpose_1/MaxMax1table_decoder/conv2d_transpose_1/BiasAdd:output:0?table_decoder/conv2d_transpose_1/Max/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2&
$table_decoder/conv2d_transpose_1/Maxñ
$table_decoder/conv2d_transpose_1/subSub1table_decoder/conv2d_transpose_1/BiasAdd:output:0-table_decoder/conv2d_transpose_1/Max:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2&
$table_decoder/conv2d_transpose_1/sub¹
$table_decoder/conv2d_transpose_1/ExpExp(table_decoder/conv2d_transpose_1/sub:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2&
$table_decoder/conv2d_transpose_1/Exp»
6table_decoder/conv2d_transpose_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ28
6table_decoder/conv2d_transpose_1/Sum/reduction_indices
$table_decoder/conv2d_transpose_1/SumSum(table_decoder/conv2d_transpose_1/Exp:y:0?table_decoder/conv2d_transpose_1/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2&
$table_decoder/conv2d_transpose_1/Sumô
(table_decoder/conv2d_transpose_1/truedivRealDiv(table_decoder/conv2d_transpose_1/Exp:y:0-table_decoder/conv2d_transpose_1/Sum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2*
(table_decoder/conv2d_transpose_1/truedivÐ
(col_decoder/conv2d/Conv2D/ReadVariableOpReadVariableOp1col_decoder_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02*
(col_decoder/conv2d/Conv2D/ReadVariableOpó
col_decoder/conv2d/Conv2DConv2Ddropout_1/dropout/Mul_1:z:00col_decoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
col_decoder/conv2d/Conv2DÆ
)col_decoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp2col_decoder_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)col_decoder/conv2d/BiasAdd/ReadVariableOpÕ
col_decoder/conv2d/BiasAddBiasAdd"col_decoder/conv2d/Conv2D:output:01col_decoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
col_decoder/conv2d/BiasAdd
col_decoder/conv2d/ReluRelu#col_decoder/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
col_decoder/conv2d/Relu
col_decoder/up_sampling2d/ShapeShape%col_decoder/conv2d/Relu:activations:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d/Shape¨
-col_decoder/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-col_decoder/up_sampling2d/strided_slice/stack¬
/col_decoder/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/col_decoder/up_sampling2d/strided_slice/stack_1¬
/col_decoder/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/col_decoder/up_sampling2d/strided_slice/stack_2ê
'col_decoder/up_sampling2d/strided_sliceStridedSlice(col_decoder/up_sampling2d/Shape:output:06col_decoder/up_sampling2d/strided_slice/stack:output:08col_decoder/up_sampling2d/strided_slice/stack_1:output:08col_decoder/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'col_decoder/up_sampling2d/strided_slice
col_decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2!
col_decoder/up_sampling2d/ConstÆ
col_decoder/up_sampling2d/mulMul0col_decoder/up_sampling2d/strided_slice:output:0(col_decoder/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
col_decoder/up_sampling2d/mul
/col_decoder/up_sampling2d/resize/ResizeBilinearResizeBilinear%col_decoder/conv2d/Relu:activations:0!col_decoder/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
half_pixel_centers(21
/col_decoder/up_sampling2d/resize/ResizeBilinear
%col_decoder/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%col_decoder/concatenate_2/concat/axis¤
 col_decoder/concatenate_2/concatConcatV2block4_pool/MaxPool:output:0@col_decoder/up_sampling2d/resize/ResizeBilinear:resized_images:0.col_decoder/concatenate_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222"
 col_decoder/concatenate_2/concat
!col_decoder/up_sampling2d_1/ShapeShape)col_decoder/concatenate_2/concat:output:0*
T0*
_output_shapes
:2#
!col_decoder/up_sampling2d_1/Shape¬
/col_decoder/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/col_decoder/up_sampling2d_1/strided_slice/stack°
1col_decoder/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1col_decoder/up_sampling2d_1/strided_slice/stack_1°
1col_decoder/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1col_decoder/up_sampling2d_1/strided_slice/stack_2ö
)col_decoder/up_sampling2d_1/strided_sliceStridedSlice*col_decoder/up_sampling2d_1/Shape:output:08col_decoder/up_sampling2d_1/strided_slice/stack:output:0:col_decoder/up_sampling2d_1/strided_slice/stack_1:output:0:col_decoder/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)col_decoder/up_sampling2d_1/strided_slice
!col_decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2#
!col_decoder/up_sampling2d_1/ConstÎ
col_decoder/up_sampling2d_1/mulMul2col_decoder/up_sampling2d_1/strided_slice:output:0*col_decoder/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d_1/mul
1col_decoder/up_sampling2d_1/resize/ResizeBilinearResizeBilinear)col_decoder/concatenate_2/concat:output:0#col_decoder/up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
half_pixel_centers(23
1col_decoder/up_sampling2d_1/resize/ResizeBilinear
%col_decoder/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%col_decoder/concatenate_3/concat/axis¦
 col_decoder/concatenate_3/concatConcatV2block3_pool/MaxPool:output:0Bcol_decoder/up_sampling2d_1/resize/ResizeBilinear:resized_images:0.col_decoder/concatenate_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2"
 col_decoder/concatenate_3/concat
!col_decoder/up_sampling2d_2/ShapeShape)col_decoder/concatenate_3/concat:output:0*
T0*
_output_shapes
:2#
!col_decoder/up_sampling2d_2/Shape¬
/col_decoder/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/col_decoder/up_sampling2d_2/strided_slice/stack°
1col_decoder/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1col_decoder/up_sampling2d_2/strided_slice/stack_1°
1col_decoder/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1col_decoder/up_sampling2d_2/strided_slice/stack_2ö
)col_decoder/up_sampling2d_2/strided_sliceStridedSlice*col_decoder/up_sampling2d_2/Shape:output:08col_decoder/up_sampling2d_2/strided_slice/stack:output:0:col_decoder/up_sampling2d_2/strided_slice/stack_1:output:0:col_decoder/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)col_decoder/up_sampling2d_2/strided_slice
!col_decoder/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2#
!col_decoder/up_sampling2d_2/ConstÎ
col_decoder/up_sampling2d_2/mulMul2col_decoder/up_sampling2d_2/strided_slice:output:0*col_decoder/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d_2/mul´
8col_decoder/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor)col_decoder/concatenate_3/concat:output:0#col_decoder/up_sampling2d_2/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
half_pixel_centers(2:
8col_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor¿
!col_decoder/up_sampling2d_3/ShapeShapeIcol_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2#
!col_decoder/up_sampling2d_3/Shape¬
/col_decoder/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/col_decoder/up_sampling2d_3/strided_slice/stack°
1col_decoder/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1col_decoder/up_sampling2d_3/strided_slice/stack_1°
1col_decoder/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1col_decoder/up_sampling2d_3/strided_slice/stack_2ö
)col_decoder/up_sampling2d_3/strided_sliceStridedSlice*col_decoder/up_sampling2d_3/Shape:output:08col_decoder/up_sampling2d_3/strided_slice/stack:output:0:col_decoder/up_sampling2d_3/strided_slice/stack_1:output:0:col_decoder/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)col_decoder/up_sampling2d_3/strided_slice
!col_decoder/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2#
!col_decoder/up_sampling2d_3/ConstÎ
col_decoder/up_sampling2d_3/mulMul2col_decoder/up_sampling2d_3/strided_slice:output:0*col_decoder/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d_3/mulÔ
8col_decoder/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborIcol_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0#col_decoder/up_sampling2d_3/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2:
8col_decoder/up_sampling2d_3/resize/ResizeNearestNeighborÁ
"col_decoder/conv2d_transpose/ShapeShapeIcol_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2$
"col_decoder/conv2d_transpose/Shape®
0col_decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0col_decoder/conv2d_transpose/strided_slice/stack²
2col_decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2col_decoder/conv2d_transpose/strided_slice/stack_1²
2col_decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2col_decoder/conv2d_transpose/strided_slice/stack_2
*col_decoder/conv2d_transpose/strided_sliceStridedSlice+col_decoder/conv2d_transpose/Shape:output:09col_decoder/conv2d_transpose/strided_slice/stack:output:0;col_decoder/conv2d_transpose/strided_slice/stack_1:output:0;col_decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*col_decoder/conv2d_transpose/strided_slice²
2col_decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2col_decoder/conv2d_transpose/strided_slice_1/stack¶
4col_decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_1/stack_1¶
4col_decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_1/stack_2
,col_decoder/conv2d_transpose/strided_slice_1StridedSlice+col_decoder/conv2d_transpose/Shape:output:0;col_decoder/conv2d_transpose/strided_slice_1/stack:output:0=col_decoder/conv2d_transpose/strided_slice_1/stack_1:output:0=col_decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,col_decoder/conv2d_transpose/strided_slice_1²
2col_decoder/conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2col_decoder/conv2d_transpose/strided_slice_2/stack¶
4col_decoder/conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_2/stack_1¶
4col_decoder/conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_2/stack_2
,col_decoder/conv2d_transpose/strided_slice_2StridedSlice+col_decoder/conv2d_transpose/Shape:output:0;col_decoder/conv2d_transpose/strided_slice_2/stack:output:0=col_decoder/conv2d_transpose/strided_slice_2/stack_1:output:0=col_decoder/conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,col_decoder/conv2d_transpose/strided_slice_2
"col_decoder/conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"col_decoder/conv2d_transpose/mul/yÐ
 col_decoder/conv2d_transpose/mulMul5col_decoder/conv2d_transpose/strided_slice_1:output:0+col_decoder/conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2"
 col_decoder/conv2d_transpose/mul
$col_decoder/conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$col_decoder/conv2d_transpose/mul_1/yÖ
"col_decoder/conv2d_transpose/mul_1Mul5col_decoder/conv2d_transpose/strided_slice_2:output:0-col_decoder/conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2$
"col_decoder/conv2d_transpose/mul_1
$col_decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$col_decoder/conv2d_transpose/stack/3°
"col_decoder/conv2d_transpose/stackPack3col_decoder/conv2d_transpose/strided_slice:output:0$col_decoder/conv2d_transpose/mul:z:0&col_decoder/conv2d_transpose/mul_1:z:0-col_decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"col_decoder/conv2d_transpose/stack²
2col_decoder/conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2col_decoder/conv2d_transpose/strided_slice_3/stack¶
4col_decoder/conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_3/stack_1¶
4col_decoder/conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_3/stack_2
,col_decoder/conv2d_transpose/strided_slice_3StridedSlice+col_decoder/conv2d_transpose/stack:output:0;col_decoder/conv2d_transpose/strided_slice_3/stack:output:0=col_decoder/conv2d_transpose/strided_slice_3/stack_1:output:0=col_decoder/conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,col_decoder/conv2d_transpose/strided_slice_3
<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpEcol_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype02>
<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp
-col_decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput+col_decoder/conv2d_transpose/stack:output:0Dcol_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0Icol_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2/
-col_decoder/conv2d_transpose/conv2d_transposeã
3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp<col_decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp
$col_decoder/conv2d_transpose/BiasAddBiasAdd6col_decoder/conv2d_transpose/conv2d_transpose:output:0;col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2&
$col_decoder/conv2d_transpose/BiasAdd³
2col_decoder/conv2d_transpose/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ24
2col_decoder/conv2d_transpose/Max/reduction_indices
 col_decoder/conv2d_transpose/MaxMax-col_decoder/conv2d_transpose/BiasAdd:output:0;col_decoder/conv2d_transpose/Max/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2"
 col_decoder/conv2d_transpose/Maxá
 col_decoder/conv2d_transpose/subSub-col_decoder/conv2d_transpose/BiasAdd:output:0)col_decoder/conv2d_transpose/Max:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2"
 col_decoder/conv2d_transpose/sub­
 col_decoder/conv2d_transpose/ExpExp$col_decoder/conv2d_transpose/sub:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2"
 col_decoder/conv2d_transpose/Exp³
2col_decoder/conv2d_transpose/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ24
2col_decoder/conv2d_transpose/Sum/reduction_indicesû
 col_decoder/conv2d_transpose/SumSum$col_decoder/conv2d_transpose/Exp:y:0;col_decoder/conv2d_transpose/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2"
 col_decoder/conv2d_transpose/Sumä
$col_decoder/conv2d_transpose/truedivRealDiv$col_decoder/conv2d_transpose/Exp:y:0)col_decoder/conv2d_transpose/Sum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2&
$col_decoder/conv2d_transpose/truediv
IdentityIdentity(col_decoder/conv2d_transpose/truediv:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity

Identity_1Identity,table_decoder/conv2d_transpose_1/truediv:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*ê
_input_shapesØ
Õ:ÿÿÿÿÿÿÿÿÿ  :::::::::::::::::::::::::::::::::::::::::::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: 

Å
$__inference_model_layer_call_fn_2876
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity

identity_1¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  *P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_27792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*ê
_input_shapesØ
Õ:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%
_user_specified_nameInput_Layer:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: 

a
(__inference_dropout_1_layer_call_fn_4076

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_21512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½

®
F__inference_block5_conv3_layer_call_and_return_conditional_losses_1602

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ç

+__inference_block2_conv2_layer_call_fn_1334

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_13242
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ë

-__inference_block_6_conv_1_layer_call_fn_1668

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_16582
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ç
`
A__inference_dropout_layer_call_and_return_conditional_losses_2116

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÇ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1763

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
|
'__inference_conv2d_1_layer_call_fn_1861

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_18512
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
©
ª
?__inference_model_layer_call_and_return_conditional_losses_3002

inputs
block1_conv1_2881
block1_conv1_2883
block1_conv2_2886
block1_conv2_2888
block2_conv1_2892
block2_conv1_2894
block2_conv2_2897
block2_conv2_2899
block3_conv1_2903
block3_conv1_2905
block3_conv2_2908
block3_conv2_2910
block3_conv3_2913
block3_conv3_2915
block3_conv4_2918
block3_conv4_2920
block4_conv1_2924
block4_conv1_2926
block4_conv2_2929
block4_conv2_2931
block4_conv3_2934
block4_conv3_2936
block4_conv4_2939
block4_conv4_2941
block5_conv1_2945
block5_conv1_2947
block5_conv2_2950
block5_conv2_2952
block5_conv3_2955
block5_conv3_2957
block5_conv4_2960
block5_conv4_2962
block_6_conv_1_2966
block_6_conv_1_2968
block_6_conv_2_2972
block_6_conv_2_2974
table_decoder_2978
table_decoder_2980
table_decoder_2982
table_decoder_2984
table_decoder_2986
table_decoder_2988
col_decoder_2991
col_decoder_2993
col_decoder_2995
col_decoder_2997
identity

identity_1¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall¢&block_6_conv_1/StatefulPartitionedCall¢&block_6_conv_2/StatefulPartitionedCall¢#col_decoder/StatefulPartitionedCall¢%table_decoder/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_2881block1_conv1_2883*
Tin
2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_12462&
$block1_conv1/StatefulPartitionedCall±
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_2886block1_conv2_2888*
Tin
2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_12682&
$block1_conv2/StatefulPartitionedCallê
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_12842
block1_pool/PartitionedCall©
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_2892block2_conv1_2894*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_13022&
$block2_conv1/StatefulPartitionedCall²
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_2897block2_conv2_2899*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_13242&
$block2_conv2/StatefulPartitionedCallë
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_13402
block2_pool/PartitionedCall©
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_2903block3_conv1_2905*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_13582&
$block3_conv1/StatefulPartitionedCall²
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_2908block3_conv2_2910*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_13802&
$block3_conv2/StatefulPartitionedCall²
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_2913block3_conv3_2915*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_14022&
$block3_conv3/StatefulPartitionedCall²
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_2918block3_conv4_2920*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_14242&
$block3_conv4/StatefulPartitionedCallé
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_14402
block3_pool/PartitionedCall§
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_2924block4_conv1_2926*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_14582&
$block4_conv1/StatefulPartitionedCall°
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_2929block4_conv2_2931*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_14802&
$block4_conv2/StatefulPartitionedCall°
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_2934block4_conv3_2936*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_15022&
$block4_conv3/StatefulPartitionedCall°
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_2939block4_conv4_2941*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_15242&
$block4_conv4/StatefulPartitionedCallé
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_15402
block4_pool/PartitionedCall§
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_2945block5_conv1_2947*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_15582&
$block5_conv1/StatefulPartitionedCall°
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_2950block5_conv2_2952*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_15802&
$block5_conv2/StatefulPartitionedCall°
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_2955block5_conv3_2957*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_16022&
$block5_conv3/StatefulPartitionedCall°
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_2960block5_conv4_2962*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_16242&
$block5_conv4/StatefulPartitionedCallé
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_16402
block5_pool/PartitionedCall±
&block_6_conv_1/StatefulPartitionedCallStatefulPartitionedCall$block5_pool/PartitionedCall:output:0block_6_conv_1_2966block_6_conv_1_2968*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_16582(
&block_6_conv_1/StatefulPartitionedCallß
dropout/PartitionedCallPartitionedCall/block_6_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_21212
dropout/PartitionedCall­
&block_6_conv_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0block_6_conv_2_2972block_6_conv_2_2974*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_16802(
&block_6_conv_2/StatefulPartitionedCallå
dropout_1/PartitionedCallPartitionedCall/block_6_conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_21562
dropout_1/PartitionedCallÑ
%table_decoder/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0table_decoder_2978table_decoder_2980table_decoder_2982table_decoder_2984table_decoder_2986table_decoder_2988*
Tin
2	*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_table_decoder_layer_call_and_return_conditional_losses_23622'
%table_decoder/StatefulPartitionedCall
#col_decoder/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0col_decoder_2991col_decoder_2993col_decoder_2995col_decoder_2997*
Tin
	2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_col_decoder_layer_call_and_return_conditional_losses_25002%
#col_decoder/StatefulPartitionedCall
IdentityIdentity,col_decoder/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall'^block_6_conv_1/StatefulPartitionedCall'^block_6_conv_2/StatefulPartitionedCall$^col_decoder/StatefulPartitionedCall&^table_decoder/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity 

Identity_1Identity.table_decoder/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall'^block_6_conv_1/StatefulPartitionedCall'^block_6_conv_2/StatefulPartitionedCall$^col_decoder/StatefulPartitionedCall&^table_decoder/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*ê
_input_shapesØ
Õ:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2P
&block_6_conv_1/StatefulPartitionedCall&block_6_conv_1/StatefulPartitionedCall2P
&block_6_conv_2/StatefulPartitionedCall&block_6_conv_2/StatefulPartitionedCall2J
#col_decoder/StatefulPartitionedCall#col_decoder/StatefulPartitionedCall2N
%table_decoder/StatefulPartitionedCall%table_decoder/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: 
Þ
|
'__inference_conv2d_2_layer_call_fn_1883

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_18732
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ê
e
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_1915

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÀ
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeBilinear
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½

®
F__inference_block4_conv2_layer_call_and_return_conditional_losses_1480

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
è
_
A__inference_dropout_layer_call_and_return_conditional_losses_4044

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
`
A__inference_dropout_layer_call_and_return_conditional_losses_4039

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÇ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

H
,__inference_up_sampling2d_layer_call_fn_1731

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_17252
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º

®
F__inference_block2_conv1_layer_call_and_return_conditional_losses_1302

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
º

ª
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1873

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp·
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
í

/__inference_conv2d_transpose_layer_call_fn_1839

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_18292
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
û
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_1340

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç

+__inference_block5_conv4_layer_call_fn_1634

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_16242
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ñ

1__inference_conv2d_transpose_1_layer_call_fn_2010

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_20002
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
²
°
G__inference_table_decoder_layer_call_and_return_conditional_losses_4374
input_0
input_1
input_2+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource
identity²
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpÁ
conv2d_1/Conv2DConv2Dinput_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_1/Conv2D¨
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp­
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_1/BiasAdd|
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_1/Relu
dropout_2/IdentityIdentityconv2d_1/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/Identity²
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOpÕ
conv2d_2/Conv2DConv2Ddropout_2/Identity:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_2/Conv2D¨
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp­
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/Reluy
up_sampling2d_4/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2®
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mulë
%up_sampling2d_4/resize/ResizeBilinearResizeBilinearconv2d_1/Relu:activations:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
half_pixel_centers(2'
%up_sampling2d_4/resize/ResizeBilineart
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisÛ
concatenate/concatConcatV2input_16up_sampling2d_4/resize/ResizeBilinear:resized_images:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
concatenate/concaty
up_sampling2d_5/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/Shape
#up_sampling2d_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_5/strided_slice/stack
%up_sampling2d_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_5/strided_slice/stack_1
%up_sampling2d_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_5/strided_slice/stack_2®
up_sampling2d_5/strided_sliceStridedSliceup_sampling2d_5/Shape:output:0,up_sampling2d_5/strided_slice/stack:output:0.up_sampling2d_5/strided_slice/stack_1:output:0.up_sampling2d_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_5/strided_slice
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_5/Const
up_sampling2d_5/mulMul&up_sampling2d_5/strided_slice:output:0up_sampling2d_5/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/mulë
%up_sampling2d_5/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
half_pixel_centers(2'
%up_sampling2d_5/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisá
concatenate_1/concatConcatV2input_26up_sampling2d_5/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
concatenate_1/concat{
up_sampling2d_6/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/Shape
#up_sampling2d_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_6/strided_slice/stack
%up_sampling2d_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_6/strided_slice/stack_1
%up_sampling2d_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_6/strided_slice/stack_2®
up_sampling2d_6/strided_sliceStridedSliceup_sampling2d_6/Shape:output:0,up_sampling2d_6/strided_slice/stack:output:0.up_sampling2d_6/strided_slice/stack_1:output:0.up_sampling2d_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_6/strided_slice
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const
up_sampling2d_6/mulMul&up_sampling2d_6/strided_slice:output:0up_sampling2d_6/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/mul
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
half_pixel_centers(2.
,up_sampling2d_6/resize/ResizeNearestNeighbor
up_sampling2d_7/ShapeShape=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d_7/Shape
#up_sampling2d_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_7/strided_slice/stack
%up_sampling2d_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_7/strided_slice/stack_1
%up_sampling2d_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_7/strided_slice/stack_2®
up_sampling2d_7/strided_sliceStridedSliceup_sampling2d_7/Shape:output:0,up_sampling2d_7/strided_slice/stack:output:0.up_sampling2d_7/strided_slice/stack_1:output:0.up_sampling2d_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_7/strided_slice
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const
up_sampling2d_7/mulMul&up_sampling2d_7/strided_slice:output:0up_sampling2d_7/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_7/mul¤
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2.
,up_sampling2d_7/resize/ResizeNearestNeighbor¡
conv2d_transpose_1/ShapeShape=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2Ô
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stack¢
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1¢
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2Þ
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_2/stack¢
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1¢
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2Þ
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y¨
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulz
conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul_1/y®
conv2d_transpose_1/mul_1Mul+conv2d_transpose_1/strided_slice_2:output:0#conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mul_1z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3ô
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/mul:z:0conv2d_transpose_1/mul_1:z:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack
(conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_3/stack¢
*conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_1¢
*conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_2Þ
"conv2d_transpose_1/strided_slice_3StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_3/stack:output:03conv2d_transpose_1/strided_slice_3/stack_1:output:03conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_3í
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpã
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transposeÅ
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpà
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/BiasAdd
(conv2d_transpose_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(conv2d_transpose_1/Max/reduction_indicesÜ
conv2d_transpose_1/MaxMax#conv2d_transpose_1/BiasAdd:output:01conv2d_transpose_1/Max/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2
conv2d_transpose_1/Max¹
conv2d_transpose_1/subSub#conv2d_transpose_1/BiasAdd:output:0conv2d_transpose_1/Max:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/sub
conv2d_transpose_1/ExpExpconv2d_transpose_1/sub:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/Exp
(conv2d_transpose_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(conv2d_transpose_1/Sum/reduction_indicesÓ
conv2d_transpose_1/SumSumconv2d_transpose_1/Exp:y:01conv2d_transpose_1/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2
conv2d_transpose_1/Sum¼
conv2d_transpose_1/truedivRealDivconv2d_transpose_1/Exp:y:0conv2d_transpose_1/Sum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/truediv|
IdentityIdentityconv2d_transpose_1/truediv:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*
_input_shapesn
l:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22:ÿÿÿÿÿÿÿÿÿdd:::::::Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/0:YU
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input/1:YU
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!
_user_specified_name	input/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ñ
ð
?__inference_model_layer_call_and_return_conditional_losses_2779

inputs
block1_conv1_2658
block1_conv1_2660
block1_conv2_2663
block1_conv2_2665
block2_conv1_2669
block2_conv1_2671
block2_conv2_2674
block2_conv2_2676
block3_conv1_2680
block3_conv1_2682
block3_conv2_2685
block3_conv2_2687
block3_conv3_2690
block3_conv3_2692
block3_conv4_2695
block3_conv4_2697
block4_conv1_2701
block4_conv1_2703
block4_conv2_2706
block4_conv2_2708
block4_conv3_2711
block4_conv3_2713
block4_conv4_2716
block4_conv4_2718
block5_conv1_2722
block5_conv1_2724
block5_conv2_2727
block5_conv2_2729
block5_conv3_2732
block5_conv3_2734
block5_conv4_2737
block5_conv4_2739
block_6_conv_1_2743
block_6_conv_1_2745
block_6_conv_2_2749
block_6_conv_2_2751
table_decoder_2755
table_decoder_2757
table_decoder_2759
table_decoder_2761
table_decoder_2763
table_decoder_2765
col_decoder_2768
col_decoder_2770
col_decoder_2772
col_decoder_2774
identity

identity_1¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall¢&block_6_conv_1/StatefulPartitionedCall¢&block_6_conv_2/StatefulPartitionedCall¢#col_decoder/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢%table_decoder/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_2658block1_conv1_2660*
Tin
2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_12462&
$block1_conv1/StatefulPartitionedCall±
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_2663block1_conv2_2665*
Tin
2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_12682&
$block1_conv2/StatefulPartitionedCallê
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_12842
block1_pool/PartitionedCall©
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_2669block2_conv1_2671*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_13022&
$block2_conv1/StatefulPartitionedCall²
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_2674block2_conv2_2676*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_13242&
$block2_conv2/StatefulPartitionedCallë
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_13402
block2_pool/PartitionedCall©
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_2680block3_conv1_2682*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_13582&
$block3_conv1/StatefulPartitionedCall²
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_2685block3_conv2_2687*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_13802&
$block3_conv2/StatefulPartitionedCall²
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_2690block3_conv3_2692*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_14022&
$block3_conv3/StatefulPartitionedCall²
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_2695block3_conv4_2697*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_14242&
$block3_conv4/StatefulPartitionedCallé
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_14402
block3_pool/PartitionedCall§
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_2701block4_conv1_2703*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_14582&
$block4_conv1/StatefulPartitionedCall°
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_2706block4_conv2_2708*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_14802&
$block4_conv2/StatefulPartitionedCall°
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_2711block4_conv3_2713*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_15022&
$block4_conv3/StatefulPartitionedCall°
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_2716block4_conv4_2718*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_15242&
$block4_conv4/StatefulPartitionedCallé
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_15402
block4_pool/PartitionedCall§
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_2722block5_conv1_2724*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_15582&
$block5_conv1/StatefulPartitionedCall°
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_2727block5_conv2_2729*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_15802&
$block5_conv2/StatefulPartitionedCall°
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_2732block5_conv3_2734*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_16022&
$block5_conv3/StatefulPartitionedCall°
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_2737block5_conv4_2739*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_16242&
$block5_conv4/StatefulPartitionedCallé
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_16402
block5_pool/PartitionedCall±
&block_6_conv_1/StatefulPartitionedCallStatefulPartitionedCall$block5_pool/PartitionedCall:output:0block_6_conv_1_2743block_6_conv_1_2745*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_16582(
&block_6_conv_1/StatefulPartitionedCall÷
dropout/StatefulPartitionedCallStatefulPartitionedCall/block_6_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_21162!
dropout/StatefulPartitionedCallµ
&block_6_conv_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0block_6_conv_2_2749block_6_conv_2_2751*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_16802(
&block_6_conv_2/StatefulPartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall/block_6_conv_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_21512#
!dropout_1/StatefulPartitionedCallÙ
%table_decoder/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0table_decoder_2755table_decoder_2757table_decoder_2759table_decoder_2761table_decoder_2763table_decoder_2765*
Tin
2	*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_table_decoder_layer_call_and_return_conditional_losses_22692'
%table_decoder/StatefulPartitionedCall
#col_decoder/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0col_decoder_2768col_decoder_2770col_decoder_2772col_decoder_2774*
Tin
	2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_col_decoder_layer_call_and_return_conditional_losses_25002%
#col_decoder/StatefulPartitionedCallà
IdentityIdentity,col_decoder/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall'^block_6_conv_1/StatefulPartitionedCall'^block_6_conv_2/StatefulPartitionedCall$^col_decoder/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall&^table_decoder/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identityæ

Identity_1Identity.table_decoder/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall'^block_6_conv_1/StatefulPartitionedCall'^block_6_conv_2/StatefulPartitionedCall$^col_decoder/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall&^table_decoder/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*ê
_input_shapesØ
Õ:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2P
&block_6_conv_1/StatefulPartitionedCall&block_6_conv_1/StatefulPartitionedCall2P
&block_6_conv_2/StatefulPartitionedCall&block_6_conv_2/StatefulPartitionedCall2J
#col_decoder/StatefulPartitionedCall#col_decoder/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2N
%table_decoder/StatefulPartitionedCall%table_decoder/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: 
½

®
F__inference_block4_conv4_layer_call_and_return_conditional_losses_1524

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ö
®
G__inference_table_decoder_layer_call_and_return_conditional_losses_2269	
input
input_1
input_2+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource
identity²
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp¿
conv2d_1/Conv2DConv2Dinput&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_1/Conv2D¨
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp­
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_1/BiasAdd|
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_1/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/dropout/Const¯
dropout_2/dropout/MulMulconv2d_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Mul}
dropout_2/dropout/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/ShapeÛ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2"
 dropout_2/dropout/GreaterEqual/yï
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dropout_2/dropout/GreaterEqual¦
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Cast«
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Mul_1²
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOpÕ
conv2d_2/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_2/Conv2D¨
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp­
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/Reluy
up_sampling2d_4/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2®
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mulë
%up_sampling2d_4/resize/ResizeBilinearResizeBilinearconv2d_1/Relu:activations:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
half_pixel_centers(2'
%up_sampling2d_4/resize/ResizeBilineart
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisÛ
concatenate/concatConcatV2input_16up_sampling2d_4/resize/ResizeBilinear:resized_images:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
concatenate/concaty
up_sampling2d_5/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/Shape
#up_sampling2d_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_5/strided_slice/stack
%up_sampling2d_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_5/strided_slice/stack_1
%up_sampling2d_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_5/strided_slice/stack_2®
up_sampling2d_5/strided_sliceStridedSliceup_sampling2d_5/Shape:output:0,up_sampling2d_5/strided_slice/stack:output:0.up_sampling2d_5/strided_slice/stack_1:output:0.up_sampling2d_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_5/strided_slice
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_5/Const
up_sampling2d_5/mulMul&up_sampling2d_5/strided_slice:output:0up_sampling2d_5/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/mulë
%up_sampling2d_5/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
half_pixel_centers(2'
%up_sampling2d_5/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisá
concatenate_1/concatConcatV2input_26up_sampling2d_5/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
concatenate_1/concat{
up_sampling2d_6/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/Shape
#up_sampling2d_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_6/strided_slice/stack
%up_sampling2d_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_6/strided_slice/stack_1
%up_sampling2d_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_6/strided_slice/stack_2®
up_sampling2d_6/strided_sliceStridedSliceup_sampling2d_6/Shape:output:0,up_sampling2d_6/strided_slice/stack:output:0.up_sampling2d_6/strided_slice/stack_1:output:0.up_sampling2d_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_6/strided_slice
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const
up_sampling2d_6/mulMul&up_sampling2d_6/strided_slice:output:0up_sampling2d_6/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/mul
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
half_pixel_centers(2.
,up_sampling2d_6/resize/ResizeNearestNeighbor
up_sampling2d_7/ShapeShape=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d_7/Shape
#up_sampling2d_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_7/strided_slice/stack
%up_sampling2d_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_7/strided_slice/stack_1
%up_sampling2d_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_7/strided_slice/stack_2®
up_sampling2d_7/strided_sliceStridedSliceup_sampling2d_7/Shape:output:0,up_sampling2d_7/strided_slice/stack:output:0.up_sampling2d_7/strided_slice/stack_1:output:0.up_sampling2d_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_7/strided_slice
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const
up_sampling2d_7/mulMul&up_sampling2d_7/strided_slice:output:0up_sampling2d_7/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_7/mul¤
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2.
,up_sampling2d_7/resize/ResizeNearestNeighbor¡
conv2d_transpose_1/ShapeShape=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2Ô
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stack¢
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1¢
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2Þ
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_2/stack¢
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1¢
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2Þ
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y¨
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulz
conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul_1/y®
conv2d_transpose_1/mul_1Mul+conv2d_transpose_1/strided_slice_2:output:0#conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mul_1z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3ô
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/mul:z:0conv2d_transpose_1/mul_1:z:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack
(conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_3/stack¢
*conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_1¢
*conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_2Þ
"conv2d_transpose_1/strided_slice_3StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_3/stack:output:03conv2d_transpose_1/strided_slice_3/stack_1:output:03conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_3í
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpã
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transposeÅ
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpà
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/BiasAdd
(conv2d_transpose_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(conv2d_transpose_1/Max/reduction_indicesÜ
conv2d_transpose_1/MaxMax#conv2d_transpose_1/BiasAdd:output:01conv2d_transpose_1/Max/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2
conv2d_transpose_1/Max¹
conv2d_transpose_1/subSub#conv2d_transpose_1/BiasAdd:output:0conv2d_transpose_1/Max:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/sub
conv2d_transpose_1/ExpExpconv2d_transpose_1/sub:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/Exp
(conv2d_transpose_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(conv2d_transpose_1/Sum/reduction_indicesÓ
conv2d_transpose_1/SumSumconv2d_transpose_1/Exp:y:01conv2d_transpose_1/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2
conv2d_transpose_1/Sum¼
conv2d_transpose_1/truedivRealDivconv2d_transpose_1/Exp:y:0conv2d_transpose_1/Sum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/truediv|
IdentityIdentityconv2d_transpose_1/truediv:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*
_input_shapesn
l:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22:ÿÿÿÿÿÿÿÿÿdd:::::::W S
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:WS
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

_user_specified_nameinput:WS
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
º

¸
*__inference_col_decoder_layer_call_fn_4181
input_0
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1input_2unknown	unknown_0	unknown_1	unknown_2*
Tin
	2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_col_decoder_layer_call_and_return_conditional_losses_25002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22:ÿÿÿÿÿÿÿÿÿdd::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/0:YU
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input/1:YU
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!
_user_specified_name	input/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Þs
É
E__inference_col_decoder_layer_call_and_return_conditional_losses_2500	
input
input_1
input_2)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource
identity¬
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp¹
conv2d/Conv2DConv2Dinput$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d/Conv2D¢
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¥
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/BiasAddv
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/Relus
up_sampling2d/ShapeShapeconv2d/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2¢
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mulã
#up_sampling2d/resize/ResizeBilinearResizeBilinearconv2d/Relu:activations:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
half_pixel_centers(2%
#up_sampling2d/resize/ResizeBilineart
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisÙ
concatenate/concatConcatV2input_14up_sampling2d/resize/ResizeBilinear:resized_images:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
concatenate/concaty
up_sampling2d_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2®
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mulë
%up_sampling2d_1/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
half_pixel_centers(2'
%up_sampling2d_1/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisá
concatenate_1/concatConcatV2input_26up_sampling2d_1/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
concatenate_1/concat{
up_sampling2d_2/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shape
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stack
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2®
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_2/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor
up_sampling2d_3/ShapeShape=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shape
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stack
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2®
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul¤
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_3/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor
conv2d_transpose/ShapeShape=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2È
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slice
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_1/stack
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2Ò
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1
&conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_2/stack
(conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_1
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_2Ò
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_2r
conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul/y 
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mulv
conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul_1/y¦
conv2d_transpose/mul_1Mul)conv2d_transpose/strided_slice_2:output:0!conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mul_1v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3è
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/mul:z:0conv2d_transpose/mul_1:z:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack
&conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_3/stack
(conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_1
(conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_2Ò
 conv2d_transpose/strided_slice_3StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_3/stack:output:01conv2d_transpose/strided_slice_3/stack_1:output:01conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_3ç
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOpÛ
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose¿
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOpØ
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose/BiasAdd
&conv2d_transpose/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&conv2d_transpose/Max/reduction_indicesÔ
conv2d_transpose/MaxMax!conv2d_transpose/BiasAdd:output:0/conv2d_transpose/Max/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2
conv2d_transpose/Max±
conv2d_transpose/subSub!conv2d_transpose/BiasAdd:output:0conv2d_transpose/Max:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose/sub
conv2d_transpose/ExpExpconv2d_transpose/sub:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose/Exp
&conv2d_transpose/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&conv2d_transpose/Sum/reduction_indicesË
conv2d_transpose/SumSumconv2d_transpose/Exp:y:0/conv2d_transpose/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2
conv2d_transpose/Sum´
conv2d_transpose/truedivRealDivconv2d_transpose/Exp:y:0conv2d_transpose/Sum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose/truedivz
IdentityIdentityconv2d_transpose/truediv:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22:ÿÿÿÿÿÿÿÿÿdd:::::W S
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:WS
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

_user_specified_nameinput:WS
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ÿ
F
*__inference_block4_pool_layer_call_fn_1546

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_15402
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_2151

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÇ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

J
.__inference_up_sampling2d_2_layer_call_fn_1769

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_17632
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

®
F__inference_block1_conv1_layer_call_and_return_conditional_losses_1246

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

J
.__inference_up_sampling2d_7_layer_call_fn_1959

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_19532
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û
a
E__inference_block5_pool_layer_call_and_return_conditional_losses_1640

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
¯
?__inference_model_layer_call_and_return_conditional_losses_2652
input_layer
block1_conv1_2531
block1_conv1_2533
block1_conv2_2536
block1_conv2_2538
block2_conv1_2542
block2_conv1_2544
block2_conv2_2547
block2_conv2_2549
block3_conv1_2553
block3_conv1_2555
block3_conv2_2558
block3_conv2_2560
block3_conv3_2563
block3_conv3_2565
block3_conv4_2568
block3_conv4_2570
block4_conv1_2574
block4_conv1_2576
block4_conv2_2579
block4_conv2_2581
block4_conv3_2584
block4_conv3_2586
block4_conv4_2589
block4_conv4_2591
block5_conv1_2595
block5_conv1_2597
block5_conv2_2600
block5_conv2_2602
block5_conv3_2605
block5_conv3_2607
block5_conv4_2610
block5_conv4_2612
block_6_conv_1_2616
block_6_conv_1_2618
block_6_conv_2_2622
block_6_conv_2_2624
table_decoder_2628
table_decoder_2630
table_decoder_2632
table_decoder_2634
table_decoder_2636
table_decoder_2638
col_decoder_2641
col_decoder_2643
col_decoder_2645
col_decoder_2647
identity

identity_1¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall¢&block_6_conv_1/StatefulPartitionedCall¢&block_6_conv_2/StatefulPartitionedCall¢#col_decoder/StatefulPartitionedCall¢%table_decoder/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_layerblock1_conv1_2531block1_conv1_2533*
Tin
2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_12462&
$block1_conv1/StatefulPartitionedCall±
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_2536block1_conv2_2538*
Tin
2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_12682&
$block1_conv2/StatefulPartitionedCallê
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_12842
block1_pool/PartitionedCall©
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_2542block2_conv1_2544*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_13022&
$block2_conv1/StatefulPartitionedCall²
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_2547block2_conv2_2549*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_13242&
$block2_conv2/StatefulPartitionedCallë
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_13402
block2_pool/PartitionedCall©
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_2553block3_conv1_2555*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_13582&
$block3_conv1/StatefulPartitionedCall²
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_2558block3_conv2_2560*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_13802&
$block3_conv2/StatefulPartitionedCall²
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_2563block3_conv3_2565*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_14022&
$block3_conv3/StatefulPartitionedCall²
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_2568block3_conv4_2570*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_14242&
$block3_conv4/StatefulPartitionedCallé
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_14402
block3_pool/PartitionedCall§
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_2574block4_conv1_2576*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_14582&
$block4_conv1/StatefulPartitionedCall°
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_2579block4_conv2_2581*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_14802&
$block4_conv2/StatefulPartitionedCall°
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_2584block4_conv3_2586*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_15022&
$block4_conv3/StatefulPartitionedCall°
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_2589block4_conv4_2591*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_15242&
$block4_conv4/StatefulPartitionedCallé
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_15402
block4_pool/PartitionedCall§
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_2595block5_conv1_2597*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_15582&
$block5_conv1/StatefulPartitionedCall°
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_2600block5_conv2_2602*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_15802&
$block5_conv2/StatefulPartitionedCall°
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_2605block5_conv3_2607*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_16022&
$block5_conv3/StatefulPartitionedCall°
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_2610block5_conv4_2612*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_16242&
$block5_conv4/StatefulPartitionedCallé
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_16402
block5_pool/PartitionedCall±
&block_6_conv_1/StatefulPartitionedCallStatefulPartitionedCall$block5_pool/PartitionedCall:output:0block_6_conv_1_2616block_6_conv_1_2618*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_16582(
&block_6_conv_1/StatefulPartitionedCallß
dropout/PartitionedCallPartitionedCall/block_6_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_21212
dropout/PartitionedCall­
&block_6_conv_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0block_6_conv_2_2622block_6_conv_2_2624*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_16802(
&block_6_conv_2/StatefulPartitionedCallå
dropout_1/PartitionedCallPartitionedCall/block_6_conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_21562
dropout_1/PartitionedCallÑ
%table_decoder/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0table_decoder_2628table_decoder_2630table_decoder_2632table_decoder_2634table_decoder_2636table_decoder_2638*
Tin
2	*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_table_decoder_layer_call_and_return_conditional_losses_23622'
%table_decoder/StatefulPartitionedCall
#col_decoder/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0col_decoder_2641col_decoder_2643col_decoder_2645col_decoder_2647*
Tin
	2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_col_decoder_layer_call_and_return_conditional_losses_25002%
#col_decoder/StatefulPartitionedCall
IdentityIdentity,col_decoder/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall'^block_6_conv_1/StatefulPartitionedCall'^block_6_conv_2/StatefulPartitionedCall$^col_decoder/StatefulPartitionedCall&^table_decoder/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity 

Identity_1Identity.table_decoder/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall'^block_6_conv_1/StatefulPartitionedCall'^block_6_conv_2/StatefulPartitionedCall$^col_decoder/StatefulPartitionedCall&^table_decoder/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*ê
_input_shapesØ
Õ:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2P
&block_6_conv_1/StatefulPartitionedCall&block_6_conv_1/StatefulPartitionedCall2P
&block_6_conv_2/StatefulPartitionedCall&block_6_conv_2/StatefulPartitionedCall2J
#col_decoder/StatefulPartitionedCall#col_decoder/StatefulPartitionedCall2N
%table_decoder/StatefulPartitionedCall%table_decoder/StatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%
_user_specified_nameInput_Layer:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: 
ç

+__inference_block3_conv3_layer_call_fn_1412

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_14022
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ã

+__inference_block1_conv1_layer_call_fn_1256

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_12462
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ç

+__inference_block3_conv1_layer_call_fn_1368

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_13582
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
è
_
A__inference_dropout_layer_call_and_return_conditional_losses_2121

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

J
.__inference_up_sampling2d_4_layer_call_fn_1902

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_18962
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
I__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_1934

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò)
¼
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1829

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ì
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3´
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpð
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Max/reduction_indices 
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Max}
subSubBiasAdd:output:0Max:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
subf
ExpExpsub:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indices
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Sum
truedivRealDivExp:y:0Sum:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
truedivy
IdentityIdentitytruediv:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
è
c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_1725

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÀ
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeBilinear
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç

+__inference_block4_conv2_layer_call_fn_1490

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_14802
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
º

ª
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1851

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp·
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
½

®
F__inference_block4_conv3_layer_call_and_return_conditional_losses_1502

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ÿ
F
*__inference_block3_pool_layer_call_fn_1446

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_14402
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç

+__inference_block3_conv4_layer_call_fn_1434

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_14242
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

B
&__inference_dropout_layer_call_fn_4054

inputs
identity¦
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_21212
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

®
F__inference_block1_conv2_layer_call_and_return_conditional_losses_1268

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

õ
?__inference_model_layer_call_and_return_conditional_losses_2528
input_layer
block1_conv1_2014
block1_conv1_2016
block1_conv2_2019
block1_conv2_2021
block2_conv1_2025
block2_conv1_2027
block2_conv2_2030
block2_conv2_2032
block3_conv1_2036
block3_conv1_2038
block3_conv2_2041
block3_conv2_2043
block3_conv3_2046
block3_conv3_2048
block3_conv4_2051
block3_conv4_2053
block4_conv1_2057
block4_conv1_2059
block4_conv2_2062
block4_conv2_2064
block4_conv3_2067
block4_conv3_2069
block4_conv4_2072
block4_conv4_2074
block5_conv1_2078
block5_conv1_2080
block5_conv2_2083
block5_conv2_2085
block5_conv3_2088
block5_conv3_2090
block5_conv4_2093
block5_conv4_2095
block_6_conv_1_2099
block_6_conv_1_2101
block_6_conv_2_2134
block_6_conv_2_2136
table_decoder_2402
table_decoder_2404
table_decoder_2406
table_decoder_2408
table_decoder_2410
table_decoder_2412
col_decoder_2517
col_decoder_2519
col_decoder_2521
col_decoder_2523
identity

identity_1¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall¢&block_6_conv_1/StatefulPartitionedCall¢&block_6_conv_2/StatefulPartitionedCall¢#col_decoder/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢%table_decoder/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_layerblock1_conv1_2014block1_conv1_2016*
Tin
2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_12462&
$block1_conv1/StatefulPartitionedCall±
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_2019block1_conv2_2021*
Tin
2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_12682&
$block1_conv2/StatefulPartitionedCallê
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_12842
block1_pool/PartitionedCall©
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_2025block2_conv1_2027*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_13022&
$block2_conv1/StatefulPartitionedCall²
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_2030block2_conv2_2032*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_13242&
$block2_conv2/StatefulPartitionedCallë
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_13402
block2_pool/PartitionedCall©
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_2036block3_conv1_2038*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_13582&
$block3_conv1/StatefulPartitionedCall²
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_2041block3_conv2_2043*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_13802&
$block3_conv2/StatefulPartitionedCall²
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_2046block3_conv3_2048*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_14022&
$block3_conv3/StatefulPartitionedCall²
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_2051block3_conv4_2053*
Tin
2*
Tout
2*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_14242&
$block3_conv4/StatefulPartitionedCallé
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_14402
block3_pool/PartitionedCall§
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_2057block4_conv1_2059*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_14582&
$block4_conv1/StatefulPartitionedCall°
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_2062block4_conv2_2064*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_14802&
$block4_conv2/StatefulPartitionedCall°
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_2067block4_conv3_2069*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_15022&
$block4_conv3/StatefulPartitionedCall°
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_2072block4_conv4_2074*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_15242&
$block4_conv4/StatefulPartitionedCallé
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_15402
block4_pool/PartitionedCall§
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_2078block5_conv1_2080*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_15582&
$block5_conv1/StatefulPartitionedCall°
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_2083block5_conv2_2085*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_15802&
$block5_conv2/StatefulPartitionedCall°
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_2088block5_conv3_2090*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_16022&
$block5_conv3/StatefulPartitionedCall°
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_2093block5_conv4_2095*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_16242&
$block5_conv4/StatefulPartitionedCallé
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_16402
block5_pool/PartitionedCall±
&block_6_conv_1/StatefulPartitionedCallStatefulPartitionedCall$block5_pool/PartitionedCall:output:0block_6_conv_1_2099block_6_conv_1_2101*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_16582(
&block_6_conv_1/StatefulPartitionedCall÷
dropout/StatefulPartitionedCallStatefulPartitionedCall/block_6_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_21162!
dropout/StatefulPartitionedCallµ
&block_6_conv_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0block_6_conv_2_2134block_6_conv_2_2136*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_16802(
&block_6_conv_2/StatefulPartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall/block_6_conv_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_21512#
!dropout_1/StatefulPartitionedCallÙ
%table_decoder/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0table_decoder_2402table_decoder_2404table_decoder_2406table_decoder_2408table_decoder_2410table_decoder_2412*
Tin
2	*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_table_decoder_layer_call_and_return_conditional_losses_22692'
%table_decoder/StatefulPartitionedCall
#col_decoder/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0col_decoder_2517col_decoder_2519col_decoder_2521col_decoder_2523*
Tin
	2*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_col_decoder_layer_call_and_return_conditional_losses_25002%
#col_decoder/StatefulPartitionedCallà
IdentityIdentity,col_decoder/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall'^block_6_conv_1/StatefulPartitionedCall'^block_6_conv_2/StatefulPartitionedCall$^col_decoder/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall&^table_decoder/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identityæ

Identity_1Identity.table_decoder/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall'^block_6_conv_1/StatefulPartitionedCall'^block_6_conv_2/StatefulPartitionedCall$^col_decoder/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall&^table_decoder/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*ê
_input_shapesØ
Õ:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2P
&block_6_conv_1/StatefulPartitionedCall&block_6_conv_1/StatefulPartitionedCall2P
&block_6_conv_2/StatefulPartitionedCall&block_6_conv_2/StatefulPartitionedCall2J
#col_decoder/StatefulPartitionedCall#col_decoder/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2N
%table_decoder/StatefulPartitionedCall%table_decoder/StatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%
_user_specified_nameInput_Layer:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: 
½

®
F__inference_block3_conv2_layer_call_and_return_conditional_losses_1380

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
½

®
F__inference_block5_conv2_layer_call_and_return_conditional_losses_1580

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
½

®
F__inference_block4_conv1_layer_call_and_return_conditional_losses_1458

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ö
À
$__inference_model_layer_call_fn_3928

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity

identity_1¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  *P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_27792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*ê
_input_shapesØ
Õ:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: 
À

°
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_1658

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp·
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
û
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_1284

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

J
.__inference_up_sampling2d_6_layer_call_fn_1940

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_19342
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
Ã
"__inference_signature_wrapper_3200
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity

identity_1¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  *P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.**
config_proto

GPU 

CPU2J 8*(
f#R!
__inference__wrapped_model_12342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*ê
_input_shapesØ
Õ:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%
_user_specified_nameInput_Layer:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: 

Å
$__inference_model_layer_call_fn_3099
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44
identity

identity_1¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  *P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_30022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*ê
_input_shapesØ
Õ:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%
_user_specified_nameInput_Layer:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: 
ê
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_4071

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
®
G__inference_table_decoder_layer_call_and_return_conditional_losses_2362	
input
input_1
input_2+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource
identity²
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp¿
conv2d_1/Conv2DConv2Dinput&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_1/Conv2D¨
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp­
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_1/BiasAdd|
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_1/Relu
dropout_2/IdentityIdentityconv2d_1/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/Identity²
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOpÕ
conv2d_2/Conv2DConv2Ddropout_2/Identity:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_2/Conv2D¨
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp­
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/Reluy
up_sampling2d_4/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2®
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mulë
%up_sampling2d_4/resize/ResizeBilinearResizeBilinearconv2d_1/Relu:activations:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
half_pixel_centers(2'
%up_sampling2d_4/resize/ResizeBilineart
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisÛ
concatenate/concatConcatV2input_16up_sampling2d_4/resize/ResizeBilinear:resized_images:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
concatenate/concaty
up_sampling2d_5/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/Shape
#up_sampling2d_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_5/strided_slice/stack
%up_sampling2d_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_5/strided_slice/stack_1
%up_sampling2d_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_5/strided_slice/stack_2®
up_sampling2d_5/strided_sliceStridedSliceup_sampling2d_5/Shape:output:0,up_sampling2d_5/strided_slice/stack:output:0.up_sampling2d_5/strided_slice/stack_1:output:0.up_sampling2d_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_5/strided_slice
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_5/Const
up_sampling2d_5/mulMul&up_sampling2d_5/strided_slice:output:0up_sampling2d_5/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/mulë
%up_sampling2d_5/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
half_pixel_centers(2'
%up_sampling2d_5/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisá
concatenate_1/concatConcatV2input_26up_sampling2d_5/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
concatenate_1/concat{
up_sampling2d_6/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/Shape
#up_sampling2d_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_6/strided_slice/stack
%up_sampling2d_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_6/strided_slice/stack_1
%up_sampling2d_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_6/strided_slice/stack_2®
up_sampling2d_6/strided_sliceStridedSliceup_sampling2d_6/Shape:output:0,up_sampling2d_6/strided_slice/stack:output:0.up_sampling2d_6/strided_slice/stack_1:output:0.up_sampling2d_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_6/strided_slice
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const
up_sampling2d_6/mulMul&up_sampling2d_6/strided_slice:output:0up_sampling2d_6/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/mul
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÈÈ*
half_pixel_centers(2.
,up_sampling2d_6/resize/ResizeNearestNeighbor
up_sampling2d_7/ShapeShape=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d_7/Shape
#up_sampling2d_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_7/strided_slice/stack
%up_sampling2d_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_7/strided_slice/stack_1
%up_sampling2d_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_7/strided_slice/stack_2®
up_sampling2d_7/strided_sliceStridedSliceup_sampling2d_7/Shape:output:0,up_sampling2d_7/strided_slice/stack:output:0.up_sampling2d_7/strided_slice/stack_1:output:0.up_sampling2d_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_7/strided_slice
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const
up_sampling2d_7/mulMul&up_sampling2d_7/strided_slice:output:0up_sampling2d_7/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_7/mul¤
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2.
,up_sampling2d_7/resize/ResizeNearestNeighbor¡
conv2d_transpose_1/ShapeShape=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2Ô
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stack¢
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1¢
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2Þ
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_2/stack¢
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1¢
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2Þ
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y¨
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulz
conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul_1/y®
conv2d_transpose_1/mul_1Mul+conv2d_transpose_1/strided_slice_2:output:0#conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mul_1z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3ô
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/mul:z:0conv2d_transpose_1/mul_1:z:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack
(conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_3/stack¢
*conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_1¢
*conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_2Þ
"conv2d_transpose_1/strided_slice_3StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_3/stack:output:03conv2d_transpose_1/strided_slice_3/stack_1:output:03conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_3í
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpã
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transposeÅ
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpà
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/BiasAdd
(conv2d_transpose_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(conv2d_transpose_1/Max/reduction_indicesÜ
conv2d_transpose_1/MaxMax#conv2d_transpose_1/BiasAdd:output:01conv2d_transpose_1/Max/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2
conv2d_transpose_1/Max¹
conv2d_transpose_1/subSub#conv2d_transpose_1/BiasAdd:output:0conv2d_transpose_1/Max:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/sub
conv2d_transpose_1/ExpExpconv2d_transpose_1/sub:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/Exp
(conv2d_transpose_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(conv2d_transpose_1/Sum/reduction_indicesÓ
conv2d_transpose_1/SumSumconv2d_transpose_1/Exp:y:01conv2d_transpose_1/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
	keep_dims(2
conv2d_transpose_1/Sum¼
conv2d_transpose_1/truedivRealDivconv2d_transpose_1/Exp:y:0conv2d_transpose_1/Sum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_transpose_1/truediv|
IdentityIdentityconv2d_transpose_1/truediv:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*
_input_shapesn
l:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22:ÿÿÿÿÿÿÿÿÿdd:::::::W S
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:WS
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

_user_specified_nameinput:WS
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ë

-__inference_block_6_conv_2_layer_call_fn_1690

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_16802
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
º
Ø
,__inference_table_decoder_layer_call_fn_4412
input_0
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_table_decoder_layer_call_and_return_conditional_losses_23622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*
_input_shapesn
l:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22:ÿÿÿÿÿÿÿÿÿdd::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/0:YU
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input/1:YU
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!
_user_specified_name	input/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "¯L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
M
Input_Layer>
serving_default_Input_Layer:0ÿÿÿÿÿÿÿÿÿ  I
col_decoder:
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ  K
table_decoder:
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿ  tensorflow/serving/predict:	
ê
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer_with_weights-15
layer-20
layer-21
layer_with_weights-16
layer-22
layer-23
layer_with_weights-17
layer-24
layer-25
layer_with_weights-18
layer-26
layer_with_weights-19
layer-27
	optimizer
loss
regularization_losses
 	variables
!trainable_variables
"	keras_api
#
signatures
Ö_default_save_signature
+×&call_and_return_all_conditional_losses
Ø__call__"
_tf_keras_modelæ{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 800, 800, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input_Layer"}, "name": "Input_Layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["Input_Layer", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv4", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv4", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv4", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv4", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv4", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv4", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block_6_conv_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block_6_conv_1", "inbound_nodes": [[["block5_pool", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["block_6_conv_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block_6_conv_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block_6_conv_2", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["block_6_conv_2", 0, 0, {}]]]}, {"class_name": "TableBranch", "config": {"name": "col_decoder", "trainable": true, "dtype": "float32"}, "name": "col_decoder", "inbound_nodes": [[["dropout_1", 0, 0, {}], ["block4_pool", 0, 0, {}], ["block3_pool", 0, 0, {}]]]}, {"class_name": "ColumnBranch", "config": {"name": "table_decoder", "trainable": true, "dtype": "float32"}, "name": "table_decoder", "inbound_nodes": [[["dropout_1", 0, 0, {}], ["block4_pool", 0, 0, {}], ["block3_pool", 0, 0, {}]]]}], "input_layers": [["Input_Layer", 0, 0]], "output_layers": [["col_decoder", 0, 0], ["table_decoder", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 800, 800, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 800, 800, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input_Layer"}, "name": "Input_Layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["Input_Layer", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv4", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv4", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv4", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv4", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv4", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv4", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block_6_conv_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block_6_conv_1", "inbound_nodes": [[["block5_pool", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["block_6_conv_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block_6_conv_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block_6_conv_2", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["block_6_conv_2", 0, 0, {}]]]}, {"class_name": "TableBranch", "config": {"name": "col_decoder", "trainable": true, "dtype": "float32"}, "name": "col_decoder", "inbound_nodes": [[["dropout_1", 0, 0, {}], ["block4_pool", 0, 0, {}], ["block3_pool", 0, 0, {}]]]}, {"class_name": "ColumnBranch", "config": {"name": "table_decoder", "trainable": true, "dtype": "float32"}, "name": "table_decoder", "inbound_nodes": [[["dropout_1", 0, 0, {}], ["block4_pool", 0, 0, {}], ["block3_pool", 0, 0, {}]]]}], "input_layers": [["Input_Layer", 0, 0]], "output_layers": [["col_decoder", 0, 0], ["table_decoder", 0, 0]]}}, "training_config": {"loss": {"table_decoder": "sparse_categorical_crossentropy", "col_decoder": "sparse_categorical_crossentropy"}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
"
_tf_keras_input_layerâ{"class_name": "InputLayer", "name": "Input_Layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 800, 800, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 800, 800, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input_Layer"}}
Î	

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
+Ù&call_and_return_all_conditional_losses
Ú__call__"§
_tf_keras_layer{"class_name": "Conv2D", "name": "block1_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 800, 800, 3]}}
Ð	

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
+Û&call_and_return_all_conditional_losses
Ü__call__"©
_tf_keras_layer{"class_name": "Conv2D", "name": "block1_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 800, 800, 64]}}
Ø
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+Ý&call_and_return_all_conditional_losses
Þ__call__"Ç
_tf_keras_layer­{"class_name": "MaxPooling2D", "name": "block1_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ñ	

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
+ß&call_and_return_all_conditional_losses
à__call__"ª
_tf_keras_layer{"class_name": "Conv2D", "name": "block2_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 400, 64]}}
Ó	

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
+á&call_and_return_all_conditional_losses
â__call__"¬
_tf_keras_layer{"class_name": "Conv2D", "name": "block2_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 400, 128]}}
Ø
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
+ã&call_and_return_all_conditional_losses
ä__call__"Ç
_tf_keras_layer­{"class_name": "MaxPooling2D", "name": "block2_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ó	

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+å&call_and_return_all_conditional_losses
æ__call__"¬
_tf_keras_layer{"class_name": "Conv2D", "name": "block3_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 200, 128]}}
Ó	

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
+ç&call_and_return_all_conditional_losses
è__call__"¬
_tf_keras_layer{"class_name": "Conv2D", "name": "block3_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 200, 256]}}
Ó	

Pkernel
Qbias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
+é&call_and_return_all_conditional_losses
ê__call__"¬
_tf_keras_layer{"class_name": "Conv2D", "name": "block3_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 200, 256]}}
Ó	

Vkernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
+ë&call_and_return_all_conditional_losses
ì__call__"¬
_tf_keras_layer{"class_name": "Conv2D", "name": "block3_conv4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block3_conv4", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 200, 256]}}
Ø
\regularization_losses
]	variables
^trainable_variables
_	keras_api
+í&call_and_return_all_conditional_losses
î__call__"Ç
_tf_keras_layer­{"class_name": "MaxPooling2D", "name": "block3_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ó	

`kernel
abias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
+ï&call_and_return_all_conditional_losses
ð__call__"¬
_tf_keras_layer{"class_name": "Conv2D", "name": "block4_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 256]}}
Ó	

fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
+ñ&call_and_return_all_conditional_losses
ò__call__"¬
_tf_keras_layer{"class_name": "Conv2D", "name": "block4_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 512]}}
Ó	

lkernel
mbias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
+ó&call_and_return_all_conditional_losses
ô__call__"¬
_tf_keras_layer{"class_name": "Conv2D", "name": "block4_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 512]}}
Ó	

rkernel
sbias
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
+õ&call_and_return_all_conditional_losses
ö__call__"¬
_tf_keras_layer{"class_name": "Conv2D", "name": "block4_conv4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block4_conv4", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 512]}}
Ø
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
+÷&call_and_return_all_conditional_losses
ø__call__"Ç
_tf_keras_layer­{"class_name": "MaxPooling2D", "name": "block4_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ó	

|kernel
}bias
~regularization_losses
	variables
trainable_variables
	keras_api
+ù&call_and_return_all_conditional_losses
ú__call__"ª
_tf_keras_layer{"class_name": "Conv2D", "name": "block5_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 512]}}
×	
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
+û&call_and_return_all_conditional_losses
ü__call__"ª
_tf_keras_layer{"class_name": "Conv2D", "name": "block5_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 512]}}
×	
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
+ý&call_and_return_all_conditional_losses
þ__call__"ª
_tf_keras_layer{"class_name": "Conv2D", "name": "block5_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 512]}}
×	
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
+ÿ&call_and_return_all_conditional_losses
__call__"ª
_tf_keras_layer{"class_name": "Conv2D", "name": "block5_conv4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block5_conv4", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 512]}}
Ü
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ç
_tf_keras_layer­{"class_name": "MaxPooling2D", "name": "block5_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ú	
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"­
_tf_keras_layer{"class_name": "Conv2D", "name": "block_6_conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block_6_conv_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 25, 512]}}
Ä
regularization_losses
	variables
 trainable_variables
¡	keras_api
+&call_and_return_all_conditional_losses
__call__"¯
_tf_keras_layer{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
Ú	
¢kernel
	£bias
¤regularization_losses
¥	variables
¦trainable_variables
§	keras_api
+&call_and_return_all_conditional_losses
__call__"­
_tf_keras_layer{"class_name": "Conv2D", "name": "block_6_conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "block_6_conv_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 25, 128]}}
È
¨regularization_losses
©	variables
ªtrainable_variables
«	keras_api
+&call_and_return_all_conditional_losses
__call__"³
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}


¬conv1
­	upsample1
®	upsample2
¯	upsample3
°	upsample4
±convtraspose
²regularization_losses
³	variables
´trainable_variables
µ	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerò{"class_name": "TableBranch", "name": "col_decoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "col_decoder", "trainable": true, "dtype": "float32"}}


¶conv1

·conv2

¸drop1
¹	upsample1
º	upsample2
»	upsample3
¼	upsample4
½convtraspose
¾regularization_losses
¿	variables
Àtrainable_variables
Á	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer÷{"class_name": "ColumnBranch", "name": "table_decoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "table_decoder", "trainable": true, "dtype": "float32"}}
"
	optimizer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper

$0
%1
*2
+3
44
55
:6
;7
D8
E9
J10
K11
P12
Q13
V14
W15
`16
a17
f18
g19
l20
m21
r22
s23
|24
}25
26
27
28
29
30
31
32
33
¢34
£35
Â36
Ã37
Ä38
Å39
Æ40
Ç41
È42
É43
Ê44
Ë45"
trackable_list_wrapper

0
1
¢2
£3
Â4
Ã5
Ä6
Å7
Æ8
Ç9
È10
É11
Ê12
Ë13"
trackable_list_wrapper
Ó
Ìlayers
regularization_losses
Ínon_trainable_variables
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
 	variables
!trainable_variables
Ø__call__
Ö_default_save_signature
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ñlayers
 Òlayer_regularization_losses
Ónon_trainable_variables
Ômetrics
&regularization_losses
Õlayer_metrics
'	variables
(trainable_variables
Ú__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ölayers
 ×layer_regularization_losses
Ønon_trainable_variables
Ùmetrics
,regularization_losses
Úlayer_metrics
-	variables
.trainable_variables
Ü__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ûlayers
 Ülayer_regularization_losses
Ýnon_trainable_variables
Þmetrics
0regularization_losses
ßlayer_metrics
1	variables
2trainable_variables
Þ__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
.:,@2block2_conv1/kernel
 :2block2_conv1/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
àlayers
 álayer_regularization_losses
ânon_trainable_variables
ãmetrics
6regularization_losses
älayer_metrics
7	variables
8trainable_variables
à__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
/:-2block2_conv2/kernel
 :2block2_conv2/bias
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ålayers
 ælayer_regularization_losses
çnon_trainable_variables
èmetrics
<regularization_losses
élayer_metrics
=	variables
>trainable_variables
â__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
êlayers
 ëlayer_regularization_losses
ìnon_trainable_variables
ímetrics
@regularization_losses
îlayer_metrics
A	variables
Btrainable_variables
ä__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
/:-2block3_conv1/kernel
 :2block3_conv1/bias
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ïlayers
 ðlayer_regularization_losses
ñnon_trainable_variables
òmetrics
Fregularization_losses
ólayer_metrics
G	variables
Htrainable_variables
æ__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
_generic_user_object
/:-2block3_conv2/kernel
 :2block3_conv2/bias
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ôlayers
 õlayer_regularization_losses
önon_trainable_variables
÷metrics
Lregularization_losses
ølayer_metrics
M	variables
Ntrainable_variables
è__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
/:-2block3_conv3/kernel
 :2block3_conv3/bias
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ùlayers
 úlayer_regularization_losses
ûnon_trainable_variables
ümetrics
Rregularization_losses
ýlayer_metrics
S	variables
Ttrainable_variables
ê__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
/:-2block3_conv4/kernel
 :2block3_conv4/bias
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
þlayers
 ÿlayer_regularization_losses
non_trainable_variables
metrics
Xregularization_losses
layer_metrics
Y	variables
Ztrainable_variables
ì__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
 layer_regularization_losses
non_trainable_variables
metrics
\regularization_losses
layer_metrics
]	variables
^trainable_variables
î__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
/:-2block4_conv1/kernel
 :2block4_conv1/bias
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
 layer_regularization_losses
non_trainable_variables
metrics
bregularization_losses
layer_metrics
c	variables
dtrainable_variables
ð__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
/:-2block4_conv2/kernel
 :2block4_conv2/bias
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
 layer_regularization_losses
non_trainable_variables
metrics
hregularization_losses
layer_metrics
i	variables
jtrainable_variables
ò__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
/:-2block4_conv3/kernel
 :2block4_conv3/bias
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
 layer_regularization_losses
non_trainable_variables
metrics
nregularization_losses
layer_metrics
o	variables
ptrainable_variables
ô__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
/:-2block4_conv4/kernel
 :2block4_conv4/bias
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
 layer_regularization_losses
non_trainable_variables
metrics
tregularization_losses
layer_metrics
u	variables
vtrainable_variables
ö__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
 layer_regularization_losses
non_trainable_variables
metrics
xregularization_losses
 layer_metrics
y	variables
ztrainable_variables
ø__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
/:-2block5_conv1/kernel
 :2block5_conv1/bias
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
¶
¡layers
 ¢layer_regularization_losses
£non_trainable_variables
¤metrics
~regularization_losses
¥layer_metrics
	variables
trainable_variables
ú__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses"
_generic_user_object
/:-2block5_conv2/kernel
 :2block5_conv2/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¦layers
 §layer_regularization_losses
¨non_trainable_variables
©metrics
regularization_losses
ªlayer_metrics
	variables
trainable_variables
ü__call__
+û&call_and_return_all_conditional_losses
'û"call_and_return_conditional_losses"
_generic_user_object
/:-2block5_conv3/kernel
 :2block5_conv3/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
«layers
 ¬layer_regularization_losses
­non_trainable_variables
®metrics
regularization_losses
¯layer_metrics
	variables
trainable_variables
þ__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
/:-2block5_conv4/kernel
 :2block5_conv4/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
°layers
 ±layer_regularization_losses
²non_trainable_variables
³metrics
regularization_losses
´layer_metrics
	variables
trainable_variables
__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
µlayers
 ¶layer_regularization_losses
·non_trainable_variables
¸metrics
regularization_losses
¹layer_metrics
	variables
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
1:/2block_6_conv_1/kernel
": 2block_6_conv_1/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
ºlayers
 »layer_regularization_losses
¼non_trainable_variables
½metrics
regularization_losses
¾layer_metrics
	variables
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¿layers
 Àlayer_regularization_losses
Ánon_trainable_variables
Âmetrics
regularization_losses
Ãlayer_metrics
	variables
 trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
1:/2block_6_conv_2/kernel
": 2block_6_conv_2/bias
 "
trackable_list_wrapper
0
¢0
£1"
trackable_list_wrapper
0
¢0
£1"
trackable_list_wrapper
¸
Älayers
 Ålayer_regularization_losses
Ænon_trainable_variables
Çmetrics
¤regularization_losses
Èlayer_metrics
¥	variables
¦trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Élayers
 Êlayer_regularization_losses
Ënon_trainable_variables
Ìmetrics
¨regularization_losses
Ílayer_metrics
©	variables
ªtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ê	
Âkernel
	Ãbias
Îregularization_losses
Ï	variables
Ðtrainable_variables
Ñ	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 25, 128]}}
©
Òregularization_losses
Ó	variables
Ôtrainable_variables
Õ	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerú{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
­
Öregularization_losses
×	variables
Øtrainable_variables
Ù	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerþ{"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¬
Úregularization_losses
Û	variables
Ütrainable_variables
Ý	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerý{"class_name": "UpSampling2D", "name": "up_sampling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¬
Þregularization_losses
ß	variables
àtrainable_variables
á	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerý{"class_name": "UpSampling2D", "name": "up_sampling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}


Äkernel
	Åbias
âregularization_losses
ã	variables
ätrainable_variables
å	keras_api
+&call_and_return_all_conditional_losses
__call__"Ô
_tf_keras_layerº{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 896}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 400, 896]}}
 "
trackable_list_wrapper
@
Â0
Ã1
Ä2
Å3"
trackable_list_wrapper
@
Â0
Ã1
Ä2
Å3"
trackable_list_wrapper
¸
ælayers
 çlayer_regularization_losses
ènon_trainable_variables
émetrics
²regularization_losses
êlayer_metrics
³	variables
´trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Î	
Ækernel
	Çbias
ëregularization_losses
ì	variables
ítrainable_variables
î	keras_api
+&call_and_return_all_conditional_losses
__call__"¡
_tf_keras_layer{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 25, 128]}}
Î	
Èkernel
	Ébias
ïregularization_losses
ð	variables
ñtrainable_variables
ò	keras_api
+&call_and_return_all_conditional_losses
__call__"¡
_tf_keras_layer{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 25, 128]}}
È
óregularization_losses
ô	variables
õtrainable_variables
ö	keras_api
+ &call_and_return_all_conditional_losses
¡__call__"³
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
­
÷regularization_losses
ø	variables
ùtrainable_variables
ú	keras_api
+¢&call_and_return_all_conditional_losses
£__call__"
_tf_keras_layerþ{"class_name": "UpSampling2D", "name": "up_sampling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
­
ûregularization_losses
ü	variables
ýtrainable_variables
þ	keras_api
+¤&call_and_return_all_conditional_losses
¥__call__"
_tf_keras_layerþ{"class_name": "UpSampling2D", "name": "up_sampling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling2d_5", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¬
ÿregularization_losses
	variables
trainable_variables
	keras_api
+¦&call_and_return_all_conditional_losses
§__call__"
_tf_keras_layerý{"class_name": "UpSampling2D", "name": "up_sampling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling2d_6", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¬
regularization_losses
	variables
trainable_variables
	keras_api
+¨&call_and_return_all_conditional_losses
©__call__"
_tf_keras_layerý{"class_name": "UpSampling2D", "name": "up_sampling2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling2d_7", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}


Êkernel
	Ëbias
regularization_losses
	variables
trainable_variables
	keras_api
+ª&call_and_return_all_conditional_losses
«__call__"Ø
_tf_keras_layer¾{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 896}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 400, 896]}}
 "
trackable_list_wrapper
P
Æ0
Ç1
È2
É3
Ê4
Ë5"
trackable_list_wrapper
P
Æ0
Ç1
È2
É3
Ê4
Ë5"
trackable_list_wrapper
¸
layers
 layer_regularization_losses
non_trainable_variables
metrics
¾regularization_losses
layer_metrics
¿	variables
Àtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
5:32col_decoder/conv2d/kernel
&:$2col_decoder/conv2d/bias
>:<2#col_decoder/conv2d_transpose/kernel
/:-2!col_decoder/conv2d_transpose/bias
9:72table_decoder/conv2d_1/kernel
*:(2table_decoder/conv2d_1/bias
9:72table_decoder/conv2d_2/kernel
*:(2table_decoder/conv2d_2/bias
B:@2'table_decoder/conv2d_transpose_1/kernel
3:12%table_decoder/conv2d_transpose_1/bias
ö
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27"
trackable_list_wrapper

$0
%1
*2
+3
44
55
:6
;7
D8
E9
J10
K11
P12
Q13
V14
W15
`16
a17
f18
g19
l20
m21
r22
s23
|24
}25
26
27
28
29
30
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Â0
Ã1"
trackable_list_wrapper
0
Â0
Ã1"
trackable_list_wrapper
¸
layers
 layer_regularization_losses
non_trainable_variables
metrics
Îregularization_losses
layer_metrics
Ï	variables
Ðtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
 layer_regularization_losses
non_trainable_variables
metrics
Òregularization_losses
layer_metrics
Ó	variables
Ôtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
 layer_regularization_losses
non_trainable_variables
metrics
Öregularization_losses
layer_metrics
×	variables
Øtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
  layer_regularization_losses
¡non_trainable_variables
¢metrics
Úregularization_losses
£layer_metrics
Û	variables
Ütrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¤layers
 ¥layer_regularization_losses
¦non_trainable_variables
§metrics
Þregularization_losses
¨layer_metrics
ß	variables
àtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
Ä0
Å1"
trackable_list_wrapper
0
Ä0
Å1"
trackable_list_wrapper
¸
©layers
 ªlayer_regularization_losses
«non_trainable_variables
¬metrics
âregularization_losses
­layer_metrics
ã	variables
ätrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
P
¬0
­1
®2
¯3
°4
±5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Æ0
Ç1"
trackable_list_wrapper
0
Æ0
Ç1"
trackable_list_wrapper
¸
®layers
 ¯layer_regularization_losses
°non_trainable_variables
±metrics
ëregularization_losses
²layer_metrics
ì	variables
ítrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
È0
É1"
trackable_list_wrapper
0
È0
É1"
trackable_list_wrapper
¸
³layers
 ´layer_regularization_losses
µnon_trainable_variables
¶metrics
ïregularization_losses
·layer_metrics
ð	variables
ñtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¸layers
 ¹layer_regularization_losses
ºnon_trainable_variables
»metrics
óregularization_losses
¼layer_metrics
ô	variables
õtrainable_variables
¡__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
½layers
 ¾layer_regularization_losses
¿non_trainable_variables
Àmetrics
÷regularization_losses
Álayer_metrics
ø	variables
ùtrainable_variables
£__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Âlayers
 Ãlayer_regularization_losses
Änon_trainable_variables
Åmetrics
ûregularization_losses
Ælayer_metrics
ü	variables
ýtrainable_variables
¥__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Çlayers
 Èlayer_regularization_losses
Énon_trainable_variables
Êmetrics
ÿregularization_losses
Ëlayer_metrics
	variables
trainable_variables
§__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ìlayers
 Ílayer_regularization_losses
Înon_trainable_variables
Ïmetrics
regularization_losses
Ðlayer_metrics
	variables
trainable_variables
©__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
Ê0
Ë1"
trackable_list_wrapper
0
Ê0
Ë1"
trackable_list_wrapper
¸
Ñlayers
 Òlayer_regularization_losses
Ónon_trainable_variables
Ômetrics
regularization_losses
Õlayer_metrics
	variables
trainable_variables
«__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
`
¶0
·1
¸2
¹3
º4
»5
¼6
½7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ë2è
__inference__wrapped_model_1234Ä
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *4¢1
/,
Input_Layerÿÿÿÿÿÿÿÿÿ  
Ê2Ç
?__inference_model_layer_call_and_return_conditional_losses_3829
?__inference_model_layer_call_and_return_conditional_losses_3525
?__inference_model_layer_call_and_return_conditional_losses_2528
?__inference_model_layer_call_and_return_conditional_losses_2652À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
$__inference_model_layer_call_fn_4027
$__inference_model_layer_call_fn_2876
$__inference_model_layer_call_fn_3928
$__inference_model_layer_call_fn_3099À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¥2¢
F__inference_block1_conv1_layer_call_and_return_conditional_losses_1246×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block1_conv1_layer_call_fn_1256×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¥2¢
F__inference_block1_conv2_layer_call_and_return_conditional_losses_1268×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2
+__inference_block1_conv2_layer_call_fn_1278×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
­2ª
E__inference_block1_pool_layer_call_and_return_conditional_losses_1284à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
*__inference_block1_pool_layer_call_fn_1290à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¥2¢
F__inference_block2_conv1_layer_call_and_return_conditional_losses_1302×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2
+__inference_block2_conv1_layer_call_fn_1312×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
¦2£
F__inference_block2_conv2_layer_call_and_return_conditional_losses_1324Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block2_conv2_layer_call_fn_1334Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
­2ª
E__inference_block2_pool_layer_call_and_return_conditional_losses_1340à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
*__inference_block2_pool_layer_call_fn_1346à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¦2£
F__inference_block3_conv1_layer_call_and_return_conditional_losses_1358Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block3_conv1_layer_call_fn_1368Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¦2£
F__inference_block3_conv2_layer_call_and_return_conditional_losses_1380Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block3_conv2_layer_call_fn_1390Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¦2£
F__inference_block3_conv3_layer_call_and_return_conditional_losses_1402Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block3_conv3_layer_call_fn_1412Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¦2£
F__inference_block3_conv4_layer_call_and_return_conditional_losses_1424Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block3_conv4_layer_call_fn_1434Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
­2ª
E__inference_block3_pool_layer_call_and_return_conditional_losses_1440à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
*__inference_block3_pool_layer_call_fn_1446à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¦2£
F__inference_block4_conv1_layer_call_and_return_conditional_losses_1458Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block4_conv1_layer_call_fn_1468Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¦2£
F__inference_block4_conv2_layer_call_and_return_conditional_losses_1480Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block4_conv2_layer_call_fn_1490Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¦2£
F__inference_block4_conv3_layer_call_and_return_conditional_losses_1502Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block4_conv3_layer_call_fn_1512Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¦2£
F__inference_block4_conv4_layer_call_and_return_conditional_losses_1524Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block4_conv4_layer_call_fn_1534Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
­2ª
E__inference_block4_pool_layer_call_and_return_conditional_losses_1540à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
*__inference_block4_pool_layer_call_fn_1546à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¦2£
F__inference_block5_conv1_layer_call_and_return_conditional_losses_1558Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block5_conv1_layer_call_fn_1568Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¦2£
F__inference_block5_conv2_layer_call_and_return_conditional_losses_1580Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block5_conv2_layer_call_fn_1590Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¦2£
F__inference_block5_conv3_layer_call_and_return_conditional_losses_1602Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block5_conv3_layer_call_fn_1612Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¦2£
F__inference_block5_conv4_layer_call_and_return_conditional_losses_1624Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_block5_conv4_layer_call_fn_1634Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
­2ª
E__inference_block5_pool_layer_call_and_return_conditional_losses_1640à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
*__inference_block5_pool_layer_call_fn_1646à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¨2¥
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_1658Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
-__inference_block_6_conv_1_layer_call_fn_1668Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
À2½
A__inference_dropout_layer_call_and_return_conditional_losses_4044
A__inference_dropout_layer_call_and_return_conditional_losses_4039´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
&__inference_dropout_layer_call_fn_4054
&__inference_dropout_layer_call_fn_4049´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_1680Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
-__inference_block_6_conv_2_layer_call_fn_1690Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ä2Á
C__inference_dropout_1_layer_call_and_return_conditional_losses_4071
C__inference_dropout_1_layer_call_and_return_conditional_losses_4066´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
(__inference_dropout_1_layer_call_fn_4076
(__inference_dropout_1_layer_call_fn_4081´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
E__inference_col_decoder_layer_call_and_return_conditional_losses_4166¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
*__inference_col_decoder_layer_call_fn_4181¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ê2Ç
G__inference_table_decoder_layer_call_and_return_conditional_losses_4374
G__inference_table_decoder_layer_call_and_return_conditional_losses_4281²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
,__inference_table_decoder_layer_call_fn_4412
,__inference_table_decoder_layer_call_fn_4393²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5B3
"__inference_signature_wrapper_3200Input_Layer
 2
@__inference_conv2d_layer_call_and_return_conditional_losses_1702Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
%__inference_conv2d_layer_call_fn_1712Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¯2¬
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_1725à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
,__inference_up_sampling2d_layer_call_fn_1731à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
±2®
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1744à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_up_sampling2d_1_layer_call_fn_1750à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
±2®
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1763à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_up_sampling2d_2_layer_call_fn_1769à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
±2®
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_1782à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_up_sampling2d_3_layer_call_fn_1788à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª2§
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1829Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_conv2d_transpose_layer_call_fn_1839Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¢2
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1851Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
'__inference_conv2d_1_layer_call_fn_1861Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¢2
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1873Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
'__inference_conv2d_2_layer_call_fn_1883Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
±2®
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_1896à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_up_sampling2d_4_layer_call_fn_1902à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
±2®
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_1915à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_up_sampling2d_5_layer_call_fn_1921à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
±2®
I__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_1934à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_up_sampling2d_6_layer_call_fn_1940à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
±2®
I__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_1953à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_up_sampling2d_7_layer_call_fn_1959à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¬2©
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2000Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
1__inference_conv2d_transpose_1_layer_call_fn_2010Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ³
__inference__wrapped_model_1234B$%*+45:;DEJKPQVW`afglmrs|}¢£ÆÇÈÉÊËÂÃÄÅ>¢;
4¢1
/,
Input_Layerÿÿÿÿÿÿÿÿÿ  
ª "ª
>
col_decoder/,
col_decoderÿÿÿÿÿÿÿÿÿ  
B
table_decoder1.
table_decoderÿÿÿÿÿÿÿÿÿ  Û
F__inference_block1_conv1_layer_call_and_return_conditional_losses_1246$%I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ³
+__inference_block1_conv1_layer_call_fn_1256$%I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Û
F__inference_block1_conv2_layer_call_and_return_conditional_losses_1268*+I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ³
+__inference_block1_conv2_layer_call_fn_1278*+I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@è
E__inference_block1_pool_layer_call_and_return_conditional_losses_1284R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
*__inference_block1_pool_layer_call_fn_1290R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÜ
F__inference_block2_conv1_layer_call_and_return_conditional_losses_130245I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ´
+__inference_block2_conv1_layer_call_fn_131245I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÝ
F__inference_block2_conv2_layer_call_and_return_conditional_losses_1324:;J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
+__inference_block2_conv2_layer_call_fn_1334:;J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿè
E__inference_block2_pool_layer_call_and_return_conditional_losses_1340R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
*__inference_block2_pool_layer_call_fn_1346R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÝ
F__inference_block3_conv1_layer_call_and_return_conditional_losses_1358DEJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
+__inference_block3_conv1_layer_call_fn_1368DEJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÝ
F__inference_block3_conv2_layer_call_and_return_conditional_losses_1380JKJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
+__inference_block3_conv2_layer_call_fn_1390JKJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÝ
F__inference_block3_conv3_layer_call_and_return_conditional_losses_1402PQJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
+__inference_block3_conv3_layer_call_fn_1412PQJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÝ
F__inference_block3_conv4_layer_call_and_return_conditional_losses_1424VWJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
+__inference_block3_conv4_layer_call_fn_1434VWJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿè
E__inference_block3_pool_layer_call_and_return_conditional_losses_1440R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
*__inference_block3_pool_layer_call_fn_1446R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÝ
F__inference_block4_conv1_layer_call_and_return_conditional_losses_1458`aJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
+__inference_block4_conv1_layer_call_fn_1468`aJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÝ
F__inference_block4_conv2_layer_call_and_return_conditional_losses_1480fgJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
+__inference_block4_conv2_layer_call_fn_1490fgJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÝ
F__inference_block4_conv3_layer_call_and_return_conditional_losses_1502lmJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
+__inference_block4_conv3_layer_call_fn_1512lmJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÝ
F__inference_block4_conv4_layer_call_and_return_conditional_losses_1524rsJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
+__inference_block4_conv4_layer_call_fn_1534rsJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿè
E__inference_block4_pool_layer_call_and_return_conditional_losses_1540R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
*__inference_block4_pool_layer_call_fn_1546R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÝ
F__inference_block5_conv1_layer_call_and_return_conditional_losses_1558|}J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
+__inference_block5_conv1_layer_call_fn_1568|}J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿß
F__inference_block5_conv2_layer_call_and_return_conditional_losses_1580J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ·
+__inference_block5_conv2_layer_call_fn_1590J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿß
F__inference_block5_conv3_layer_call_and_return_conditional_losses_1602J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ·
+__inference_block5_conv3_layer_call_fn_1612J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿß
F__inference_block5_conv4_layer_call_and_return_conditional_losses_1624J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ·
+__inference_block5_conv4_layer_call_fn_1634J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿè
E__inference_block5_pool_layer_call_and_return_conditional_losses_1640R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
*__inference_block5_pool_layer_call_fn_1646R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿá
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_1658J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¹
-__inference_block_6_conv_1_layer_call_fn_1668J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿá
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_1680¢£J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¹
-__inference_block_6_conv_2_layer_call_fn_1690¢£J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
E__inference_col_decoder_layer_call_and_return_conditional_losses_4166ÙÂÃÄÅ¢
¢

*'
input/0ÿÿÿÿÿÿÿÿÿ
*'
input/1ÿÿÿÿÿÿÿÿÿ22
*'
input/2ÿÿÿÿÿÿÿÿÿdd
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ  
 û
*__inference_col_decoder_layer_call_fn_4181ÌÂÃÄÅ¢
¢

*'
input/0ÿÿÿÿÿÿÿÿÿ
*'
input/1ÿÿÿÿÿÿÿÿÿ22
*'
input/2ÿÿÿÿÿÿÿÿÿdd
ª ""ÿÿÿÿÿÿÿÿÿ  Û
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1851ÆÇJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
'__inference_conv2d_1_layer_call_fn_1861ÆÇJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÛ
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1873ÈÉJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
'__inference_conv2d_2_layer_call_fn_1883ÈÉJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÙ
@__inference_conv2d_layer_call_and_return_conditional_losses_1702ÂÃJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ±
%__inference_conv2d_layer_call_fn_1712ÂÃJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿä
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2000ÊËJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¼
1__inference_conv2d_transpose_1_layer_call_fn_2010ÊËJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿâ
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1829ÄÅJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 º
/__inference_conv2d_transpose_layer_call_fn_1839ÄÅJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿµ
C__inference_dropout_1_layer_call_and_return_conditional_losses_4066n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 µ
C__inference_dropout_1_layer_call_and_return_conditional_losses_4071n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
(__inference_dropout_1_layer_call_fn_4076a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ
(__inference_dropout_1_layer_call_fn_4081a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿ³
A__inference_dropout_layer_call_and_return_conditional_losses_4039n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ³
A__inference_dropout_layer_call_and_return_conditional_losses_4044n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
&__inference_dropout_layer_call_fn_4049a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ
&__inference_dropout_layer_call_fn_4054a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿ±
?__inference_model_layer_call_and_return_conditional_losses_2528íB$%*+45:;DEJKPQVW`afglmrs|}¢£ÆÇÈÉÊËÂÃÄÅF¢C
<¢9
/,
Input_Layerÿÿÿÿÿÿÿÿÿ  
p

 
ª "_¢\
UR
'$
0/0ÿÿÿÿÿÿÿÿÿ  
'$
0/1ÿÿÿÿÿÿÿÿÿ  
 ±
?__inference_model_layer_call_and_return_conditional_losses_2652íB$%*+45:;DEJKPQVW`afglmrs|}¢£ÆÇÈÉÊËÂÃÄÅF¢C
<¢9
/,
Input_Layerÿÿÿÿÿÿÿÿÿ  
p 

 
ª "_¢\
UR
'$
0/0ÿÿÿÿÿÿÿÿÿ  
'$
0/1ÿÿÿÿÿÿÿÿÿ  
 ¬
?__inference_model_layer_call_and_return_conditional_losses_3525èB$%*+45:;DEJKPQVW`afglmrs|}¢£ÆÇÈÉÊËÂÃÄÅA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "_¢\
UR
'$
0/0ÿÿÿÿÿÿÿÿÿ  
'$
0/1ÿÿÿÿÿÿÿÿÿ  
 ¬
?__inference_model_layer_call_and_return_conditional_losses_3829èB$%*+45:;DEJKPQVW`afglmrs|}¢£ÆÇÈÉÊËÂÃÄÅA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "_¢\
UR
'$
0/0ÿÿÿÿÿÿÿÿÿ  
'$
0/1ÿÿÿÿÿÿÿÿÿ  
 
$__inference_model_layer_call_fn_2876ßB$%*+45:;DEJKPQVW`afglmrs|}¢£ÆÇÈÉÊËÂÃÄÅF¢C
<¢9
/,
Input_Layerÿÿÿÿÿÿÿÿÿ  
p

 
ª "QN
%"
0ÿÿÿÿÿÿÿÿÿ  
%"
1ÿÿÿÿÿÿÿÿÿ  
$__inference_model_layer_call_fn_3099ßB$%*+45:;DEJKPQVW`afglmrs|}¢£ÆÇÈÉÊËÂÃÄÅF¢C
<¢9
/,
Input_Layerÿÿÿÿÿÿÿÿÿ  
p 

 
ª "QN
%"
0ÿÿÿÿÿÿÿÿÿ  
%"
1ÿÿÿÿÿÿÿÿÿ  
$__inference_model_layer_call_fn_3928ÚB$%*+45:;DEJKPQVW`afglmrs|}¢£ÆÇÈÉÊËÂÃÄÅA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "QN
%"
0ÿÿÿÿÿÿÿÿÿ  
%"
1ÿÿÿÿÿÿÿÿÿ  
$__inference_model_layer_call_fn_4027ÚB$%*+45:;DEJKPQVW`afglmrs|}¢£ÆÇÈÉÊËÂÃÄÅA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "QN
%"
0ÿÿÿÿÿÿÿÿÿ  
%"
1ÿÿÿÿÿÿÿÿÿ  Å
"__inference_signature_wrapper_3200B$%*+45:;DEJKPQVW`afglmrs|}¢£ÆÇÈÉÊËÂÃÄÅM¢J
¢ 
Cª@
>
Input_Layer/,
Input_Layerÿÿÿÿÿÿÿÿÿ  "ª
>
col_decoder/,
col_decoderÿÿÿÿÿÿÿÿÿ  
B
table_decoder1.
table_decoderÿÿÿÿÿÿÿÿÿ  ­
G__inference_table_decoder_layer_call_and_return_conditional_losses_4281áÆÇÈÉÊË¢
¢

*'
input/0ÿÿÿÿÿÿÿÿÿ
*'
input/1ÿÿÿÿÿÿÿÿÿ22
*'
input/2ÿÿÿÿÿÿÿÿÿdd
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ  
 ­
G__inference_table_decoder_layer_call_and_return_conditional_losses_4374áÆÇÈÉÊË¢
¢

*'
input/0ÿÿÿÿÿÿÿÿÿ
*'
input/1ÿÿÿÿÿÿÿÿÿ22
*'
input/2ÿÿÿÿÿÿÿÿÿdd
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ  
 
,__inference_table_decoder_layer_call_fn_4393ÔÆÇÈÉÊË¢
¢

*'
input/0ÿÿÿÿÿÿÿÿÿ
*'
input/1ÿÿÿÿÿÿÿÿÿ22
*'
input/2ÿÿÿÿÿÿÿÿÿdd
p
ª ""ÿÿÿÿÿÿÿÿÿ  
,__inference_table_decoder_layer_call_fn_4412ÔÆÇÈÉÊË¢
¢

*'
input/0ÿÿÿÿÿÿÿÿÿ
*'
input/1ÿÿÿÿÿÿÿÿÿ22
*'
input/2ÿÿÿÿÿÿÿÿÿdd
p 
ª ""ÿÿÿÿÿÿÿÿÿ  ì
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1744R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_up_sampling2d_1_layer_call_fn_1750R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1763R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_up_sampling2d_2_layer_call_fn_1769R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_1782R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_up_sampling2d_3_layer_call_fn_1788R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_1896R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_up_sampling2d_4_layer_call_fn_1902R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_1915R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_up_sampling2d_5_layer_call_fn_1921R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_1934R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_up_sampling2d_6_layer_call_fn_1940R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_1953R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_up_sampling2d_7_layer_call_fn_1959R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_1725R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_up_sampling2d_layer_call_fn_1731R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ