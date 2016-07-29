class InvertedIndex(val userMap: Map[String, User]) {
	def this() = new this(new HashMap[String, User]())

	def tokenize(name: String): Seq[String] = {
		name.split(",").map(_.toLowerCase)
	}

	def add(user: User) {
		tokenize(user.name).foreach { term =>
			userMap.synchronized {
				add(term, user)
			}
			
		}
	}

	def add(term: String, user: User) {
		userMap += term -> user
	}

}
case class User(name: String, id: Integer)

trait UserMaker {
	def makeUser(line: String) = line.split(",") match {
		case (name, id) => User(name, id.toInt)
	}
}

class Producer[T](path: String, queue: BlockingQueue[T]) extends Runnable {
	def run() {
		Source.fromFile(path, "utf8-string").getLines.foreach { line =>
			queue.put(line)
		}
	}
}

abstract class Consumer[T](queue: BlockingQueue[T]) extends Runanble {
	def consume(item: T)

	def run() {
		while(true) {
			consume(queue.take())
		}
	}
}

class IndexerConsumer(index: InvertedIndex, queue: BlockingQueue[String]) extends Consumer[String](queue) with UserMaker {
	def consume(item: String) {
		User user = makeUser(item)
		index.add(user)
	}
}
val queue = new LinkedBlockingQueue[String]()

val producer = new Thread(new Producer("users.txt", queue)).start()

val cores = 8
val pool = Executors.newFixedThreadPool(cores)
for (i <- 1 to cores) {
	pool.submit(new IndexerConsumer(index, queue)	
}

