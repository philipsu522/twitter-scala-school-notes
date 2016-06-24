//// Expressions:
1 + 1

//// Values - cannot change:
val two = 1 + 1

//// Variables - can chagne:
var name = "Steve"
name = "Marius"

//// Loops
// for loops
// Numeric range: <starting integer> [to|until] <ending integer> [by increment]
// for (<identifier> <- <iterator>) [yield] [<expression>]
for(x <- 1 to 7) { println(x) }
// while loops
// while (<Boolean expression>) statement

//// Functions
def addOne(m: Int): Int = m + 1
// Don't need parenthesis if no args
def three() = 1 + 2
three()
three

// Functional programming - functions should be first-class
// not only declared and invoked but can be used in every
// segment of the language as another data type
// 1. May be created in literal form without ever having been assigned an identifier
// 2. May be stored in a container such as a value, variable or data structure
// 3. May be used as a parameter to another function or used as the return value from antoher func

def double(x: Int): Int = x * 2
// Need to specify function type or use wildcard for function value (myDouble)
val myDouble: (Int) => Int = double
val myDouble2 = double _
def max(a: Int, b: Int) = if (a > b) a else b
val maximize: (Int, Int) => Int = max

// Higher-order functions - functions that accept other functions as parameters
// and/or use functions as return values (e.g. map() and reduce())
def safeStringOp(s: String, f: String => String) = {
	if (s != null) f(s) else s
}
def reverser(s: String) = s.reverse
safeStringOp(null, reverser)
safeStringOp("Ready", reverser)

// Function literal - a working function that lacks a name
/*
You can think of a function literal as being to a function value 
what a string literal (e.g., “Hello, World”) is to a string value: 
a literal expression of the assigned data.
*/
//// Anonymous Functions
(x: Int) => x + 1
val addOne = (x: Int) => x + 1
// Use {} for more space
def timesTwo(i: Int) = {
	println("Hello World")
	i * 2
}
// {} with anonymous functions
{i: Int =>
	println("hello world")
	i * 2
}

//// Partial Application
def adder(m: Int, n: Int) = m + n
val add2 = adder(2, _: Int)
add2(3)

//// Curried Functions
def multiply(m: Int)(n: Int): Int = m * n
multiply(2)(3)
val timesTwo = multiply(2)_
timesTwo(3)
// Curry function of multiple arguments
val curriedAdd = (adder _).curried
val addTwo = curriedAdd(2)
addTwo(4)
// Invoking method with infix notation vs operator notation
val d = 65.642
d.+(2.721)
d + 2.721
//// Variable length arguments
def capitalizeAll(args: String*) = {
	args.map { arg =>
		arg.capitalize
	}
}

// Placeholder - reduces amount of extra code required to call first-class functions
def combination(x: Int, y: Int, f: (Int, Int) => Int) = f(x,y)
combination(23, 12, _ * _)
// Placeholder with type parameters
def combination[A, B](a: A, b: A, f: (A, A) => B) = f(a,b)
tripleOp[Int, Int](23, 92, _ + _)
tripleOp[Int, Boolean](23, 92, 23 > 92)

//// Classes
class Calculator {
	val brand: String = "HP"
	def add(m: Int, n: Int) = m + n
}

class Calculator(brand: String) {
	val color: String = if (brand == "TI") {
		"blue"
	} else if (brand == "HP") {
		"black"
	} else {
		"white"
	}
	def add(m: Int, n: Int) = m + n
}
// By adding "val" or "var" before a class parameter, it becomes a field in the class
class User(val name: String) {
	override def toString = s"User($name)"
}

class ScientificCalculator(brand: String) extends Calculator(brand) {
	def log(m: Double, base: Double) = math.log(m) / math.log(base)
}

class EvenMoreScientificCalculator(brand: String) extends ScientificCalculator(brand) {
	def log(m: Int): Double = log(m, math.exp(1))
}

abstract class Shape {
	def getArea():Int
}

class Circle(r: Int) extends Shape {
	def getArea(): Int = {r * r * 3}
}
val s = new Shape // Throws an error
val c = new Circle(2)


//// Traits
trait Car {
	val brand: String
}

trait Shiny {
	val shineRefraction: Int
}

class BMW extends Car {
	val brand = "BMW"
}
// Multiple traits "with"
class BMW extends Car with Shiny {
	val brand = "BMW"
	val shineRefraction = 12
}


//// Types - generic types
trait Cache[K, V] {
	def get(key: K) : V
	def put(key: K, value: V)
	def delete(key: K)
}
// Type parameters in methods
def remove[K](key: K)


//// Apply Methods - when a class or object only has one use
class Foo {}
object FooMaker {
	def apply() = new Foo
}

val newFoo = FooMaker()
class Bar {
	def apply() = 0
}
val bar = new Bar
bar()


//// Objects - used to hold single instances of a class, often for factories (singletons)
object Timer {
	var count = 0
	def currentCount(): Long {
		count += 1
		count
	}
}
Timer.currentCount()
// Companion Object
class Bar(foo: String)
object Bar {
	def apply(foo: String) = new Bar(foo)
}


//// Functions are Objects
// Int => Int
object addOne extends Function1[Int, Int] {
	def apply(m: Int): Int = m + 1
}
// Now we can call the object like a function
addOne(1)
// Classes can extend Function and those instances can be called with ()
class AddOne extends Function1[Int, Int] {
	def apply(m: Int): Int = m + 1
}
val plusOne = new AddOne()
plusOne(1)
// Shorthand for Function1
class addOne extends (Int => Int) {
	def apply(m: Int): Int = m + 1
}


//// Packages - every line in Scala REPL is its own package
// Values and functions cannot be oustide of a class or object. 
// Objects are a useful tool for organizing static functions
object colorHolder {
	val BLUE = "Blue"
	val RED = "Red"
}
// println("the color is: " + com.twitter.example.coloHolder.BLUE)


//// Pattern Matching
val times = 1
times match {
	case 1 => "one"
	case 2 => "two"
	case _ => "some other number"
}
// Matching with guards
times match {
	case i if i == 1 => "one"
	case i if i == 2 => "two"
	case _ => "some other number"
}
// Matching on type
def bigger(o: Any): Any = {
	o match {
		case i: Int if i < 0 => i - 1
		case i: Int => i + 1
		case d: Double if d < 0.0 => d - 0.1
		case d: Double => d + 0.1
		case text: String => text + "s"
	}
}
// Matching on class members
def calcType(calc: Calculator) = calc match {
	case _ if calc.brand == "hp" && calc.model == "20B" => "financial"
	case _ if calc.brand == "hp" && calc.model == "48G" => "scientific"
	case _ if calc.brand == "hp" && calc.model == "30B" => "business"
	case _ => "unknown"
}
// Case Classes - above is too painful
// automatically have equality and nice toString
case class Calculator(brand: String, model: String)
val hp20b = Calculator("hp", "20b")
val hp20B = Calculator("hp", "20b")
hp20b == hp20B
// Case Classes with pattern matching
val hp20b = Calculator("hp", "20B")
val hp30b = Calculator("hp", "30B")
def calcType(calc: Calculator) = calc match {
	case Calculator("hp", "20B") => "financial"
	case Calculator("hp", "48G") => "scientific"
	case Calculator("hp", "30B") => "business"
	case Calculator(ourBrand, ourModel) => "Calculator: %s %s is of unknown type".format(ourBrand, ourModel)
}


//// Exceptions
try {
	remoteCalculatorService.add(1,2)
} catch {
	case e: ServerIsDownException => log.error(e, "the remote call service is unavailable")
	case _: => log.error(_)
} finally {
	remoteCalculatorService.close()
}
// expression-oriented try
val result: Int = try {
	remoteCalculatorService.add(1,2)
} catch {
	case e: ServerIsDownException => {
		log.error(e)
		0
	}
} finally {
	remoteCalculatorService.close()
}


//// Data Structures
// Immutable subtypes: List, Set and Map (collection.immutable)
// Mutable: Buffer, Set and Map (collection.mutable)
// Lists
val numbers = List(1,2,3,4)
// Sets
Set(1,1,2) // Set(1,2)
// Seq - a defined order (trait)
Seq(1,1,2)
// Tuples
val hostPort = ("localhost", 80)
// Tuple accessors - 0 based
hostPort._1
hostPort._2
// Tuple pattern matching
hostPort match {
	case ("localhost", port) => "localhost"
	case (host, port) => "unknown"
}
// Tuple of 2 values
1->2
// Maps
Map(1 -> 2)
Map("foo" -> "bar")
// Maps to Maps or functions as values
Map(1 -> Map("foo" -> "bar"))
Map("timesTwo" -> {timesTwo(_)})
// Options - containers that may or may not hold something
// Interface for Option
trait Option[T] {
	def isDefined: Boolean
	def get: T
	def getOrElse(t: T): T
}
// Option has 2 subclasses: Some[T] & None
val numbers = Map("one" -> 1, "two" -> 2)
val result = numbers.get("three").getOrElse(0) * 2
// or, alternatively use pattern matching
val result = numbers.get("three") match {
	case Some(n) => n * 2
	case None => 0
}
// Streams
def inc(i: Int): Stream[Int] = Stream.cons(i, inc(i + 1))
val l = s.take(5).toList
def to(head: Char, end: Char): Stream[Char] = (head > end) match {
	case true => Stream.empty
	case false => head #:: to((head + 1).toChar, end)
}
// HashMap (mutable) - .getOrElseUpdate, +=
val numbers = collection.mutable.Map(1->2)
numbers.get(1)
numbers += (4->1)
// Other mutable: ListBuffer, ArrayBuffer, LinkedList, DoubleLinkedList,
// PriorityQueue, Stack, ArrayStack, StringBuilder

///// Collections
// Traversing list
var i = List(1,2,3)
def visit(i: List[Int]) { 
	while(i != Nil) {
		print(i.head + ", ")
		visit(i.tail)
	}
}

//// Functional Combinators
// Applies the function squared 
List(1,2,3) map squared
numbers = List(1,2,3,4)
// map 
// evaluates a function over each element in the list, 
// returning a list with the same number of elements
numbers.map((i: Int) => i * 2)
// foreach 
// like map but returns nothing (used for side-effects)
numbers.foreach((i: Int) => i * 2)
// filter 
// removes any elements where the function you pass in evaluates to false
// functions that return a Boolean are often called predicate functions
numbers.filter((i: Int) => i % 2 == 0)
numbers.filter(_ % 2 == 0)
// zip
// aggregates the contents of 2 lists into a single list of pairs
// gives List((1,a), (2,b), (3,c))
List(1,2,3).zip(List("a", "b", "c"))
// partition
// splits a list based on where it falls with respect to predicate func
val numbers = List(1,2,3,4,5,6,7,8,9,10)
numbers.partition(_ % 2 == 0)
// find
// returns first element in collection that matches a predicate func
numbers.find((i: Int) => i > 5)
// drop
// drops the first i elements
numbers.drop(5)
// dropWhile
// removes the first element(s) that match a predicate func
numbers.dropWhile(_ % 2 != 0)
// foldLeft
// 0 is starting value, m acts as an accumulator
numbers.foldLeft(0)((m: Int, n: Int) => m + n)
// foldRight runs in opposite direction, n acts as an accumulator
// flatten collapses one level of nested structure
List(List(1,2), List(3,4)).flatten
// List(1,2,3,4)
// flatMap
// combines mapping and flattening
// takes a func that works on a nested list then concatenates the results together
val nestedNumbers = List(List(1,2), List(3,4))
nestedNumbers.flatMap(x => x.map(_ * 2))
// it is shorthand for:
nestedNumbers.map((x: List[Int]) => x.map(_ * 2)).flatten


// Generalized functional combinators
// every functional combaintor above can be written on top of fold
def ourMap(numbers: List[Int], fn: Int => Int): List[Int] = {
	numbers.foldRight(List[Int]()) { 
		(x: Int, xs: List[Int]) => fn(x) :: xs
	}
}
// Note: (::) prepends a single item where (:::) prepends a complete list


// All functional combinators work on Maps too (list of tuples)
extensions = Map("steve" -> 100, "bob" -> 101, "joe" -> 201)
extensions.filter((namePhone: (String, Int)) => namePhone._2 < 200)
// Cleaner way is to use pattern match
extensions.filter( {case (name, extension) => extension < 200})
// Note: This works because filter expects a function
// In this case it is a predicate function of (PhoneExt) => Boolean
// A PartialFunction is a subtype of Function


//// Function Composition
def f(s: String) = "f(" + s + ")"
def g(s: String) = "g(" + s + ")"
// compose makes a new function that composes other functions
val fComposeG = f _ compose g _ // f(g(x)), or f(_) compose g(_)
// andThen is like compose but calls first and then second
val fAndThenG = f _ andThen g _


//// PartialFunction
val one: PartialFunction[Int, String] = {case 1 => "one"}
one.isDefinedAt(1) // true
one.isDefinedAt(2) // false
// applying the partial function
one(1)
// PartialFunctions composed with something new using orElse
val two: PartialFunction[Int, String] = {case 2 => "two"}
val three: PartialFunction[Int, String] = {case 3 => "three"}
val wildcard: PartialFunction[Int, String] = {case _ => "something else"}
val partial = one orElse two orElse three orElse wildcard



//// Types in Scala (features)
// parametric polymorphism: roughly, generic programming
// (local) type inference roughly, why you needn't say val i: Int = 12: Int
// existential qualifcation roughly, defining something for some unnamed type
// views roughly, "castability" of values of one type to another

// Variance
// covariant: C[T'] is a subclass of C[T] => [+T]
// contravariant: C[T] is a sublcass of C[T'] => [-T]
// invariant: C[T] and C[T'] are not related => [T]
class Covariant[+A]
val cv: Covariant[AnyRef] = new Covariant[String]
class Contravariant[-A]
val cv: Contravariant[String] = new Covariant[AnyRef]
class Animal { val sound = "rustle" }
class Bird extends Animal { override val sound = "call" }
class Chicken extends Bird { override val sound = "cluck" }
// Function that needs a bird but takes animal - contravariance
val getTweet: (Bird => String) = ((a: Animal) => a.sound )
// Function that returns a bird but have a chicken - covariance
val hatch: (() => Bird) = (() => new Chicken )

class Car { override def toString = "Car()" }
class Volvo extends Car { override def toString = "Volvo()" }
// Valid:
val c: Car = new Volvo()
case class Item[A](a: A) { def get: A = a}
val c: Item[Car] = new Item[Volvo](new Volvo) // this breaks, thats why we use +A
// Covariance is a great tool for morphing type parameters into their base types
// An input parameter for a method cannot be covariant because this would mean 
// it would be bound to a subtype but invoked with a base type. This is
// impossible because a base type cannot be converted to a subtype
class Car; class Volvo extends Car; class VolvoWagon extends Volvo
// Covariant - if A is Volvo, get() will return Volvo
// which we should be able to store in Car
class Item[+A](a: A) { def get: A = a } 
// Contravariant - if A is Volvo, check() takes a Volvo
// which we should be able to pass in VolvoWagon
class Check[-A] { def check(a: A) = {} }
def item(v: Item[Volvo]) { val c: Car = v.get }
def check(v: Check[Volvo]) { v.check(new VolvoWagon()) }
item(new Item[Car](new Car())) // error - parameter can morph from a subclass to a base class only
check (new Check[VolvoWagon]()) // error - parameter cannot move from a base class to a subclass

// Function[-T1 +R] are contravariant in parameter T1, covariant in parameter R
// A Function1[Any, T] is safe to use as a Function1[String, T]
// Similarly, a Function1[T, String] is safe to use as a Function1[T, Any]

// Bounds
// Upper bounds (<: - "is a")
def cacophony[T <: Animal](things: Seq[T]) things map (_.sound)
// Lower bounds: List defines ::[B >: T](x: B)
// where B >: T specifies type B as a superclass of T
val flock = List(new Bird, new Bird) // List[Bird]
new Animal :: flock // List[Animal] = List(Animal, Bird, Bird)

// Quantification - sometimes we do not care about naming a type variable
def count[A](l: List[A]) = l.size
// is equivalent to
def count(l: List[_]) = l.size
// which is shorthand for
def count(l: List[T forSome {type T}]) = l.size
// But you can lose information
def drop1(l: List[_]) = l.tail // drop1: (List[_])List[Any]
// Applying wildcard type variables
def hashcodes(l: Seq[_ <: AnyRef]) = l map (_.hashCode)
hashcodes(Seq("one", "two", "three"))


// Advanced Types
// View Bounds ("type classes")
// specifies a type that can be "viewed as" another - makes sense
// for an operation that needs to "read" an object but not modify

// Implicit functions allow automatic conversion
// allow on-demand function application when this can help satisfy type inference
implicit def strToInt(x: String) = x.toInt
val y: Int = "123"
math.max("123", 111)

// View bounds (<% - "can be seen as") (like type bounds) demand such a function exists for the given type
class Container[A <% Int] { def addIt(x:A) = 123 + x }
(new Container[String]).addIt("123") // Int: 246
def f[A <% Ordered[A]](a: A, b: A) = if (a < b) a else b // Ordered[A] defines: <(other: A)

// Implicit parameters
// Scala's math library defines an implicit Numeric[T],
// then List's definition uses it
sum[B >: A](implicit num: Numeric[B]): B
// Example:
case class Alcohol(liters: Double)
case class Water(liters: Double)

case class Fire(heat: Double)
trait Flammable[A] {
	def burn(fuel: A): Fire
}

implicit object AlocholIsFlammable extends Flammable[Alcohol] {
	def burn(fuel: Alcohol) = Fire(120.0)
}
def setFire[T](fuel: T)(implicit f: Flammable[T]) = f.burn(fuel)
setFire(Alcohol(1.0)) //ok
setFire(Water(1.0)) // fail

def pp(implicit i: Int, a: Long) = println(i,a)
implicit def v = 7
implicit val x = 10L
pp // prints (7,10)

// "Evidence" for a type without setting up objects like Numeric
A =:= B // equal
A <:< B // subtype
A <%< B // viewable as
class Container[A](value: A) { def addIt(implicit evidence: A <%< Int) = 123 + value}


// Generic programming with views
// views are primarily used to implement generic functions over collections
def min[B >: A](implicit cmp: Ordering[B]): A = {
	if (isEmpty)
		throw new UnsupportedOperationException("empty.min")
	reduceLeft((x,y) => if (cmp.lteq(x, y)) x else y)
}
// Can define your own orderings without any additional library support
List(1,2,3,4).min(new Ordering[Int] { def compare(a: Int, b: Int) b compare a })
// Essentially, Ordered is like Comparable, compares "this" vs that
// while Ordering is like Comparator and compares 2 parameter values a, b
// Conversion between the 2
implicit def ordered[A <: Ordered[A]]: Ordering[A] = new Ordering[A] {
	def compare(x: A, y: A) = x.compare(y)
}

// Context bounds (: - "has a") & implicitly
def foo[A](implicit x: Ordered[A]){}
def f[A: Ordering](a: A, b: A) = implicitly[Ordering[A]].compare(a,b) // sugar
// which is equivalent to
def f[A](a: A, b: A)(implicit ord: Ordering[A]) = ord.compare(a,b) // de-sugared

// Higher-kinded types & ad-hoc polymorphism
trait Container[M[_]] { 
	def put[A](x: A): M[A]; 
	def get[A](m: M[A]): A
}
val container = new Container[List] { 
	def put[A](x: A) = List(x);
	def get[A](m: List[A]) = m.head;
}
container.put("hey")
container.put(123)
// ad-hoc polymorphism: the ability to write generic functions over containers
implicit val listContainer = new Container[List] {
	def put[A](x: A) = List(x)
	def get[A](m: List[A]) = m.head
}
implicit val optionContainer = new Container[Some] {
	def put[A](x: A) = Some(x)
	def get[A](m: List[Some]) = m.get
}
def tupelize[M[_]: Container, A, B](fst: M[A], snd: M[B]) = {
	val c = implicitly[Container[M]]
	c.put(c.get(fst, c.get(snd)))
}
tupelize(Some(1), Some(2)) // Some[(Int, Int)] = Some((1,2))
tupelize(List(1), List(2)) // List[(Int, Int)] = List((1,2))

// F-bounded polymorphism - "recursive"
trait Container extends Ordered[Container]
// this necessitates: def compare(that: Container): Int
class MyContainer extends Container {
	def compare(that: MyContainer): Int // throws an error b/c we are specifying Ordered for
	// Container, not the particular subtype
}
// So we must use F-bounded polymorphism
trait Container[A <: Container[A]] extends Ordered[A]
// Now, Ordered is parametized on A which itself is Container[A]
class MyContainer extends Container {
	def compare(that: MyContainer) = 0
}
List(new MyContainer, new MyContainer).min

// Structural types - type requirements are expressed by interface structure instead of a concrete type (duck)
def foo(x: {def get: Int}) = 123 + x.get
foo(new { def get = 10 })

// Abstract type members - you can leave type members abstract in traits
trait Foo {
	type A;
	val x: A;
	def getX: A = x;
}
(new Foo { type = Int, val = 123}).getX
// Good for dependency injection - a service is made part of the dependent client's state
// The client delegates to external code (the injector) the responsibility of providing its 
// dependencies. The client is not allowed to call the injector code. It is the injecting 
// code that constructs the services and calls the client to inject them. This means the 
// client code does not need to know about the injecting code. The client does not need to 
// know how to construct the services. The client does not need to know which actual services 
// it is using. The client only needs to know about the intrinsic interfaces of the services 
// because these define how the client may use the services. This separates the responsibilities 
// of use and construction.

// Manifests - type erasure in Java, so it's a way to recover type information
def name[T](implicit m: scala.reflect.Manifest[T]) = m.toString

//. Converting between Java and Scala collection types - asScala, asJava
import scala.collection.JavaConverters._
val sl = new scala.collection.mutable.ListBuffer[Int]
val jl: java.util.List[Int] = sl.asJava
val sl2: scala.collection.mutable.Buffer[Int] = jl.asScala
assert(sl eq sl2)


//// Testing with Specification
import org.specs._
object ArithmeticSpec extends Specification {
	"Arithmetic" should {
		"add two numbers" in {
			1 + 1 mustEqual 2
		}
		"add three numbers" in {
			1 + 1 + 1 mustEqual 3
		}
	}
}
// 1 mustEqual 1 is a common placeholder "expectation"
// Duplication - "add", nest expectations
object ArithmeticSpec extends Specification {
	"Arithmetic" should {
		"add" in {
			"two numbers" in {
				1 + 1 mustEqual 2
			}
			"three numbers" in {
				1 + 1 + 1 mustEqual 3
			}
		}
	}
}

// Execution Model
object ExecSpec extends Specification {
	"Mutations are isolated" should {
		var x = 0
		"x equals 1 if we set it." in {
			x = 1
			x mustEqual 1
		}
		"x is the default value if we don't change it" in {
			x mustEqual 0
		}
	}
}

// Setup, Teardown 
// doBefore, doAfter
"my system" should {
	doBefore { resetTheSystem() /** user-defined reset func */}
	"mess up the system" in {...}
	"and again" in {...}
	doAfter { cleanThingsUp() }
}
// doFirst, doLast - single-time setup
"Foo" should {
	doFirst{ openTheCurtains() }
	"test stateless methods" in {...}
	"test other stateless methods" in {...}
	doLast{ closeTheCurtains() }
}

// Matchers - making sure data is right
// mustEqual - reference equality, value equality
"a" mustEqual "a"

// elements in a Sequence
val numbers = List(1, 2, 3)
numbers must contain(1)
numbers must not contain(4)
numbers must containAll(List(1, 2, 3))
numbers must containInOrder(List(1,2,3))
List(1, List(2, 3, List(4)), 5) must haveTheSameElementsAs(List(5, List(4),2,3), 1)

// items in a Map
map must haveKey(k)
map must notHaveKey(k)
map must haveValue(v)
map must notHaveValue(v)

// Numbers
a must beGreaterThan(b)
a must beGreaterThanOrEqualTo(b)
a must beLessThan(b)
a must beLessThanOrEqualTo(b)
a must beCloseTo(b, delta)

// Options
a must beNone
a must beSome[Type]
a must beSomething
a must beSome(value)

// throwA
a must throwA[WhateverException]
a must throwA(WhateverException("message"))
// Match on the exception
a must throwA(new Exception) like {
	case Exception(m) => m.startsWith("bad")
}

// Writing your own Matchers
import org.specs.matcher.Matcher
"A matcher" should {
	"be created as a val" in {
		val beEven = new Matcher[Int] {
			def apply(n: Int) = { // success, success msg, fail, fail message
				(n % 2 == 0, "%d is even".format(n), "%d is odd".format(n))
			}
		}
		2 must beEven
	}
}
// As a case class
case class beEven(b: Int) extends Matcher[Int]() {
	def apply(n: Int) = (n % 2 == 0, "%d is even".format(n), "%d is odd".format(n))
}

// Mocks
import org.specs.Specification
import org.specs.mock.Mockito

class Foo[T] {
	def get(i: Int): T
}

object MockExampleSpec extends Specification with Mockito {
	val m = mock[Foo[String]]

	m.get(0) returns "one"

	m.get(0)

	there was one(m).get(0)

	there was no(m).get(1)
}
// Spies
val list = new LinkedList[String]
val spiedList = spy(list)
spiedList.size returns 100
spiedList.add("one")
spiedList.add("two")


//// Concurrency
// Runnable/Callable
trait Runnable {
	def run(): Unit
}
// Callable is similar except it returns a value
trait Callable[V] {
	def call(): V
}

// Threads
// takes a Runnable and call "start" to run the Runnable
val hello = new Thread( new Runnable {
	def run() {
		println("hello world")
	}
})
hello.start
// ExecutorService has a variety of policies such as thread pooling

import java.net.{Socket, ServerSocket}
import java.util.concurrent.{Executors, ExecutorService}
import java.util.Date

class NetworkService(port: Int, poolSize: Int) extends Runnable {
	val serverSocket = new ServerSocket(port)
	val pool: ExecutorService = Executors.newFixedThreadPool(poolSize)

	def run() {
		try {
			while (true) {
				// This blocks
				val socket = serverSocket.accept()
				// (new Thread((new Handler(socket))).start()
				pool.execute(new Handler(socket))
			}
			} finally {
				pool.shutdown()
			}
	}
}

class Handler(socket: Socket) extends Runnable {
	def message = (Thread.currentThread.getName() + "\n").getBytes

	def run() {
		socket.getOutputStream.write(message)
		socket.getOutputStream.close()
	}
}
(new NetworkService(2020, 2)).run

// Futures - an asynchronous computation
// wrap a computation in a future, and when you need a result
// call a blocking get(). Executor returns a Future.
// A FutureTask is a Runnable and is designed to be run by an Executor
val future = new FutureTask[String](new Callable[String]() {
	def call(): String = {
		searcher.search(target);
	}
})
executor.execute(future)
// Now, need the results
val blockingResult = future.get()

// Thread Safety
// Synchronization - mutex
class Person(var name: String) {
	def set(changedName: String) {
		this.synchronized {
			name = changedName
		}
	}
}

// Volatile - similar to synchronized except nulls allowed
class Person(@volatile var name: String) {
	def set(changedName: String) {
		name = changedName
	}
}

// AtomicReference
import java.util.concurrent.atomic.AtomicReference

class Person(val name: AtomicReference[String]) {
	def set(changedName: String) {
		name.set(changedName)
	}
}

// CountDownLatch - allows multiple threads to communicate
val doneSignal = new CountDownLatch(2)
doAsyncWork(1)
doAsyncWork(2)
doneSignal.await()
println("both workers finished!")

// AtomicInteger/Long, AtomicBoolean, ReadWriteLock

// Search Engine
import scala.collection.mutable

case class User(name: String, id: Int)

class InvertedIndex(val userMap: mutable.Map[String, User]) {
	def this() = this(new mutable.HashMap[String, User])

	def tokenizeName(name: String): Seq[String] = {
		name.split(" ").map(_.toLowerCase)
	}

	def add(term: String, user: User) {
		userMap += term -> user
	}

	def add(user: User) {
		// tokenizeName(user.name).foreach{ term =>
		// 	add(term, user)
		// }
		// tokenizeName is expensive operation
		val tokens = tokenizeName(user.name)
		tokens.foreach{ term =>
			userMap.synchronized {
				add(term, user)
			}
		}
		
	}
}

// Synchronized map
import scala.collection.mutable.SynchronizedMap

class SynchronizedInvertedIndex(userMap: mutable.Map[String, User]) extends InvertedIndex(userMap) {
	def this() = this(new mutable.HashMap[String, User] with SynchronizedMap[String, User])
}

// ConcurrentHashMap
import java.util.concurrent.ConcurrentHashMap
import scala.collection.JavaConverters._
class ConcurrentInvertedIndex(userMap: collection.mutable.ConcurrentHashMap[String, User]) 
	extends InvertedIndex(userMap){
		def this() = this(new ConcurrentHashMap[String, User] asScala)
}

// Loading our InvertedIndex
trait UserMaker {
	def makeUser(line: String) = line.split(",") match {
		case Array(name, userid) => User(name, userid.trim().toInt)
	}
}

class FileRecordProducer(path: String) extends UserMaker {
	def run() {
		Source.fromFile(path, "utf-8").getLines.foreach { line =>
			index.add(makeUser(line))
		}
	}
}

// SOLUTION: Producer-Consumer
import java.util.concurrent.{BlockingQueue, LinkedBlockingQueue}

// Concrete producer
class Producer[T](path: String, queue: BlockingQueue[T]) extends Runnable {
	def run() {
		Source.fromFile(path, "utf-8").getLines.foreach{ line =>
			queue.put(line)
		}
	}
}

// Abstract consumer
abstract class Consumer[T](queue: BlockingQueue[T]) extends Runnable {
	def run() {
		while (true) {
			val item = queue.take()
			consume(item)
		}
	}

	def consume(x: T)
}

val queue = new LinkedBlockingQueue[String]()

// One thread for the producer
val producer = new Producer[String]("users.txt", queue)
new Thread(producer).start()

trait UserMaker {
	def makeUser(line: String) = line.split(",") match {
		case Array(name, userid) => User(name, userid.trim().toInt)
	}
}

class IndexerConsumer(index: InvertedIndex, queue: BlockingQueue[String]) extends Consumer[String](queue) with UserMaker {
	def consume(t: String) = index.add(makeUser(t))
}

// Let's pretend we have 8 cores on this machine
val cores = 8
val pool = Executres.newFixedThreadPool(cores)

// Submit one consumer per core
for (i <- 1 to cores) {
	pool.submit(new IndexConsumer[String](index, queue))
}


//// Java interoperability
// Classes
import java.io.IOException
import scala.throws
import scala.reflect.{BeanProperty, BooleanBeanProperty}

class SimpleClass(name: String, val acc: String, @BeanPropery var mutable: String) {
	val foo = "foo"
	val bar = "bar"
	@BeanProperty
	val fooBean = "foobean"
	@BeanProperty
	var barBean = "barbean"
	@BooleanBeanProperty
	var awesome = true

	def dangerFoo() = {
		throw new IOException("SURPRISE!")
	}

	@throws(classOf[IOException])
	def dangerBar() = {
		throw new IOException("No SURPRISE!")
	}
}
// vals can access via foo()

// vars get a method _$eq defined: foo$_eq("newfoo")

// @BeanProperty generates getters/setters that look like POJO
// so we can use setFoo("newfoo") and getFoo()

// Can't do this in java because Scala doesn't have checked exceptions
try {
	s.dangerFoo();
} catch( IOException e) {
	// UGLY
}
// Instead, use the throws annotation

// Traits
trait MyTrait {
	def traitName: String // abstract
	def upperTraitName = traitName.toUpperCase // implemented
}

public class JTraitImpl implements MyTrait {
	private String name = null;

	public JTraitImpl(String name) {
		this.name = name
	}

	public String traitName() {
		return name;
	}

	// javap says this is abstract, delegate this call to the generated
	// Scala implementation
	public String upperTraitName() {
		return MyTrait$class.upperTraitName(this);
	}
}

// Objects - scala objects compiled to a class that has a trailing "$"
class TraitImpl(name: String) extends MyTrait {
	def traitName = name
}

object TraitImpl { // Companion object
	def apply() = new TraitImpl("foo")
	def apply(name: String) = new TraitImpl(name)
}

MyTrait foo = TraitImpl$.MODULE$.apply("foo")
// Or use forwarding methods
MyTrait foo = TraitImpl.apply("foo")

// Closure function - return value depends on one or more variables declared
// outside the function
val multiplier = (i: Int) => i * factor
class ClosureClass {
	def printResult[T](f: => T) = { // Function0
		println(f)
	}

	def printResult[T](f: String => T) = { // FUnction1
		println(f("HI THERE"))
	}
}

val cc = new ClosureClass
cc.printResult { "HI MOM" }



//// Finagle
// Futures
// Finagle uses com.twitter.util.Future
val myFuture = MySlowService(request) // returns right away
// ... do other things ...
val serviceResult = mFuture.get() // blocks until service "fills in" myFuture

// Usually use "callbacks"
val future = dispatch(req)
future onSuccess { reply =>
	println(reply)
}
// Promise is a concrete subclass of the abstract Future class

// Sequential Composition
// flatMap sequences two futures, takes a Future and an async
// function and returns another Future
// use flatMap when the next thing returns a Future
def Future[A].flatMap[B](f: A => Future[B]): Future[B]
// Example
class User(n: String) { val name = n }
def isBanned(u: User) = { Future.value(false) }
val pru = new Promise[User]
val futBan = pru flatMap isBanned
futBan.get() // This hangs
pru.setValue(new User("prudence"))
futBan.get() // false

// map to apply a synchronous function to a Future
class RawCredentials(u: String, pw: String) {
	val username = u
	val password = pw
}
class Credentials(u: String, pw: String) {
	val username = u
	val password = pw
}
def normalize(raw: RawCredentials) = {
	new Credentials(raw.username.toLowerCase(), raw.password)
}
val praw = new Promise[RawCredentials]
val fcred = praw map normalize
fcred.get() // hangs
praw.setValue(new RawCredentials("Florence", "nightingale"))
praw.get().username // florence

// Syntactic shorthand to invoke flatMap
// Same example - async login & async isBanned check
def authenticate(req: LoginRequest) = {
	Future.value(new User(req.username))
}
val f = for {
	u <- authenticate(request)
	b <- isBanned(u)
} yield(u, b)
// where f: Future[(User, Boolean)]

// Concurrent Composition
// Might want to fetch data from more than one service at once
// convert a sequence of Future into a Future of sequence
// esentailly package several Futures into a single Future
object Future {
	//...
	def collect[A](fs: Seq[Future[A]]): Future[Seq[A]]
	def join(fs: Seq[Future[_]]): Future[Unit]
	def select(fs: Seq[Future[A]]): Future[(Try[A], Seq[Future[A]])]
}
// collect takes a set of Futures of the same type and yields a Future
// of a sequence of values of that type
val f2 = Future.value(2)
val f3 = Future.value(3)
val f23 = Future.collect(Seq(f2, f3))
val f5 = f23 map (_.sum) // call sum on the seq in the Future
f5.get() // 5

// join takes a sequence of Futures whose types may be mixed, yielding a Future[Unit]
// that is complete when all the underlying Futures are
val ready = Future.join(Seq(f2, f3))
ready.get() // no return value but I know my futures are done

// select returns a Future that is complete when the first of the given Futures
// complete. It returns that Future together with a Seq containing the 
// remaining uncompleted Futures
val pr7 = new Promise[Int]
val sel = Future.select(Seq(f2, pr7))
val(complete, stragglers) = sel.get()
complete.get() // 2
stragglers(0).get() // hangs
pr7.setValue(7)
stragglers(0).get() // 7

// Service
// a service that handles RPCs, taking requests and giving back replies
abstract class Service[-Req, +Rep] extends (Req => Future[Rep])
// client and servers defined in terms of Services

// client "imports" a Service from the Network, has 2 parts
// - A function to use the Service: dispatch a Req and handle a Future[Rep]
// - Configuration of how to dispatch requests; e.g. as HTTP requests to port 80 of api.twitter.com
// server "exports" a Service to the network, has 2 parts
// - A function to implement the service: take a Req and return a Future[Rep]
// Configuration of how to "listen" for incoming Reqs; e.g. as HTTP requests on port 80
// "filters" sit between services, modifying data that flows through it
// Client implementation:
val client: Service[HttpRequest, HttpResponse] = ClientBuilder()
	.codec(Http())
	.hosts("twitter.com:80")
	.hostConnectionLimit(1)
	.build()

val req = new DefaultHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/")

val f = client(req) // Client, send the request

// Handle the response
f onSuccess { res =>
	println("got response", res)
} onFailure { exc =>
	println("failed :-(", exc)
}
// Server implementation // OK resposne for root, 404 for other paths
// define our service
val rootService = new Service[HttpRequest, HttpResponse] {
	def apply(request: HttpRequest) = {
		val r = request.getUri match {
			case "/" => new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK)
			case _ => new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.NOT_FOUND)
		}
		Future.value(r)
	}
}

// Serve our service on a port
val address: SocketAddress = new InetSocketAddress(10000)
val server: Server = ServerBuilder()
	.codec(Http())
	.bindTo(address)
	.name("HttpServer")
	.build(rootService)

// Filters - transform services. They can provide service generic functionality
// For example, many services that should support rate limiting - write a filter
// that does rate-limiting and apply it to all services
// A simple proxy (rewriteReq and rewriteRes can provide protocol translation)
class MyService(client: Service[...]) extends Service[HttpRequest, HttpResponse] {
	def apply(request: HttpRequest) = {
		client(rewriteReq(request)) map { res =>
			rewriteRes(res)
		}
	}
}

abstract class Filter[-ReqIn, +RepOut, +ReqOut, -RepIn] 
	extends ((ReqIn, Service[ReqOut, RepIn]) => Future[RepOut])

//    ((ReqIn, Service[ReqOut, RepIn])
//         => Future[RepOut])
//            (*   Service   *)
// [ReqIn -> (ReqOut -> RepIn) -> RepOut]

// Filters compose together with andThen
// Providing a Service as an argument to andThen creates a (filtered) Service
val authFilter: Filter[HttpReq, HttpRep, AuthHttpReq, HttpRep]
val timeoutFilter[Req, Rep]: Filter[Req, Rep, Req, Rep]
val serviceRequiringAuth: Service[AuthHttpReq, HttpRep]

val authenticateAndTimedOut: Filter[HttpReq, HttpRep, AuthHttpReq, HttpRep] =
	authFilter andThen timeoutFilter

val authenticatedTimedOutService: Service[HttpReq, HttpRep] =
	authenticateAndTimedOut andThen serviceRequiringAuth

// Builders
// A ClientBUilder produces a Service instance given a set of parameters
// A ServerBuilder takes a Service instance and dispatches incoming requests on it
// To determine type of Service, must provide a Codec - the underlying protocol
// implementation (e.g. HTTP, thrift, memcached)

// A client that load balances over 3 given hosts, establishes at most 1 conn per host,
// and gives up after 2 failures. Stats reported to ostrich
val client: Service[HttpRequest, HttpResponse] = ClientBuilder()
	.codec(Http)
	.hosts("host1.twitter.com:10000,host2.twitter.com:10001,host3.twitter.com:10003")
	.hostConnectionLimit(1)
	.tcpConnectTimeout(1.second)
	.retries(2)
	.reportTo(new OstrichStatsReceiver)
	.build()

// A server that lists on port, a Thrift server which dispatches requests to service,
// each connection allowed up to 5 mins, a requests must be sent within 2 minutes
val service = new MyService(...) // construct instance of Finagle service
val filter = new MyFilter(...) // maybe some filters
val filteredService = filter andThen service
val server = ServerBuilder()
	.bindTo(new InetSocketAddress(port))
	.codec(ThriftServerFramedCodec())
	.name("my filtered service")
	.hostConnectionMaxLifeTime(5.minutes)
	.readTimeout(2.minutes)
	.build(filteredService)






