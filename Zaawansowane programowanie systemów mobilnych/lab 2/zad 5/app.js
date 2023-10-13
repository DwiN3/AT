function getMails(collections) {
  const pattern = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g;
  
  const mails = collections
    .filter((item) => typeof item === 'string')
    .map((str) => str.match(pattern))
    .flat()
  
  const filteredMails = mails.filter((mail) => mail);
  const sortedMails = [...new Set(filteredMails)].sort();
  return sortedMails
}

const collections = [
  {},
  15,
  "hello@test.pl",
  null,
  ['aaa', 'bbb', 5],
  'admin@gmail.com',
  undefined,
  'a34@yahoo.com',
  '321@a',
  '321.pl'
];

const Mails = getMails(collections);
console.log(Mails);
