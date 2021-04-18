export interface Tweet {
  id: number;
  tweet_id: string;
  source: number;
  tweet: string;
}

export interface FilteredTweet {
  id: number;
  tweet_id: string;
  source: number;
  tweet: string;
  category: number;
}

export interface Category {
  id: number;
  category_name: string;
}
