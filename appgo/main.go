package main

import (
	"appgo/lib"
	"fmt"
	"log"
	"os"

	"github.com/jinzhu/gorm"
	_ "github.com/jinzhu/gorm/dialects/postgres"
)

type DBInfo struct {
	Host     string
	Port     string
	User     string
	DBName   string
	Password string
}

func (info *DBInfo) String() string {
	return fmt.Sprintf("host=%s port=%s user=%s dbname=%s password=%s sslmode=disable",
		info.Host, info.Port, info.User, info.DBName, info.Password)
}

// NewDBInfo creates DBInfo from environment variables.
func NewDBInfo() *DBInfo {
	return &DBInfo{
		Host:     os.Getenv("POSTGRES_HOST"),
		Port:     os.Getenv("POSTGRES_PORT"),
		User:     os.Getenv("POSTGRES_USER"),
		DBName:   os.Getenv("POSTGRES_DB"),
		Password: os.Getenv("POSTGRES_PASSWORD"),
	}
}

func main() {
	db, err := gorm.Open("postgres", NewDBInfo().String())
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	db.AutoMigrate(&lib.User{})

	user := lib.User{
		Name:   "otomakorihciok",
		Age:    34,
		Gender: "Men",
	}

	if err := db.Create(&user).Error; err != nil {
		log.Fatal(err)
	}
}
