"""Sqlalchemy."""

from os import name
from user import User, Base

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(
    'postgresql+psycopg2://postgres:postgres@localhost:5432/testdb')
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)


def main():
  session = Session()
  user = User(name='ko', fullname='otomakorihciok', nickname='otom')
  session.add(user)
  session.query(User).filter_by(name='ko').first()
  session.commit()


if __name__ == '__main__':
  main()
