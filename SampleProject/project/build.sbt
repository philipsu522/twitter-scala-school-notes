libraryDependencies ++= Seq(
"org.scala-tools.testing" % "specs_2.10" % "1.6.+" % "test",
"org.codehaus.jackson" % "jackson-core-asl" % "1.9.+"
)

publishTo := Some(Resolver.file("file", new File(Path.userHome.absolutePath+"/.m2/repository")))

lazy val print = taskKey[Unit]("Print test message")

print := streams.value.log.info("a test message")